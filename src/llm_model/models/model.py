import torch
import torch.nn as nn
from transformers import AutoConfig,  AutoTokenizer, AutoModelForCausalLM
from llm_model.layers.StandardNorm import Normalize
from llm_model.layers.mlp import MLP
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        configs.llm_ckp_dir = 'your_path'
        self.llama_config = AutoConfig.from_pretrained(configs.llm_ckp_dir)
        self.llama = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=True,
            config=self.llama_config,
            torch_dtype=torch.float16 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            configs.llm_ckp_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
       
        self.hidden_dim_of_llama =self.llama_config.hidden_size
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_llama, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
           
            self.decoder = MLP(self.hidden_dim_of_llama, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 
    def _encode_header(self, role):
        tokens = []
        tokens.append(self.tokenizer.get_vocab()["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(role))
        tokens.append(self.tokenizer.get_vocab()["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n"))
        return tokens
    def _encode_content(
        self, content):
        tokens = self._encode_header('user')
    
        tokens.extend(
                    self.tokenizer(
        content,
        add_special_tokens=False,
        return_attention_mask=False
    )['input_ids'][0]
                )

        tokens.append(
            self.tokenizer.get_vocab()["<|eot_id|>"]
        )
        return tokens

    def format_dialog(self, prompt):
        tokens = []
        tokens.append(self.tokenizer.get_vocab()["<|begin_of_text|>"])

        toks = self._encode_content(prompt)
        tokens.extend(toks)

        tokens.extend(self._encode_header('assistant'))

        return tokens
    def LLMout(self, prompt) :
        prompt = "User: "+prompt[0]+'\nAssistant:'
        print('input ',prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llama.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # decode output
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        response_text = response[len(prompt):].strip()
        return response_text
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bs, T, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        times_embeds = self.encoder(fold_out)  # [bs*n_vars x token_num x hidden_dim]
 
        if self.mix:
            x_mark_enc = x_mark_enc.repeat_interleave(3, dim=0) 
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        
        # Processing with LLAMA
        outputs = self.llama.model(inputs_embeds=times_embeds)[0]
        
        # Reorganize shapes to separate different variables
        outputs = outputs.reshape(bs, n_vars, token_num, -1)
        
        dec_out = self.decoder(outputs)  # [bs x token_num x token_len]
        
        # Adjusting the output dimension
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out[:, :, 0:1]
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)