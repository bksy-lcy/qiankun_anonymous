
import numpy as np
import fixed_env as env
import sys

# Converting a string back to a list
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
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
MINIMUM_BUFFER_S = 10
BUFFER_TARGET_S = 30
gp = 1 - 0 + (np.log(VIDEO_BIT_RATE[-1] / float(VIDEO_BIT_RATE[0])) - 0) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # log
vp = MINIMUM_BUFFER_S/(0+ gp -1)

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM


    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,buffer_init=buffer_init,video_init= video_init)


    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch, video_batch, rebuf_batch,smooth_batch = [],[],[],[]

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

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward); video_batch.append(VIDEO_BIT_RATE[bit_rate] / M_IN_K);rebuf_batch.append( REBUF_PENALTY * rebuf);smooth_batch.append( SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K)
        # print(VIDEO_BIT_RATE[bit_rate], rebuf)
        last_bit_rate = bit_rate

 

        score = -65535
        for q in range(len(VIDEO_BIT_RATE)):
            s = (vp * (np.log(VIDEO_BIT_RATE[q] / float(VIDEO_BIT_RATE[0])) + gp) - buffer_size) / next_video_chunk_sizes[q]
            # s = (vp * (VIDEO_BIT_RATE[q]/1000. + gp) - buffer_size) / next_video_chunk_sizes[q] # lin
            if s>=score:
                score = s
                bit_rate = q

        bit_rate = int(bit_rate)

        if end_of_video:

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here


            
            video_count += 1


        if end_of_bw:
            print(r_batch[1:], video_batch[1:], rebuf_batch[1:],smooth_batch[1:],buffer_size, 48-video_chunk_remain)
            break


if __name__ == '__main__':
    main()
