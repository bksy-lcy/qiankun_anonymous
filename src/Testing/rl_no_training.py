import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import fixed_env as env
import a3c
import sys
def str_to_list(s):
    s = s.strip('[]')
    return [float(x) for x in s.split(',')]

all_cooked_bw = str_to_list(sys.argv[1])
all_cooked_time = str_to_list(sys.argv[2])
buffer_init = float(sys.argv[3])
video_init = int(sys.argv[4])

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
NN_MODEL = './models/pretrain_linear_reward.ckpt'

log_file = open('rl.log','w')
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,buffer_init=buffer_init,video_init= video_init)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch, video_batch, rebuf_batch,smooth_batch = [],[],[],[]
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, end_of_bw = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward); video_batch.append(VIDEO_BIT_RATE[bit_rate] / M_IN_K);rebuf_batch.append( REBUF_PENALTY * rebuf);smooth_batch.append( SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K)
            last_bit_rate = bit_rate

            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
            log_file.flush()
 

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here


                
                video_count += 1


            if end_of_bw:
                print(r_batch[1:], video_batch[1:], rebuf_batch[1:],smooth_batch[1:],buffer_size, 48-video_chunk_remain)
                break

             

if __name__ == '__main__':
    main()
