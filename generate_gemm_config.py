import subprocess 


def get_bert_gemm():
    import os.path as osp

    cur_path = osp.abspath('.')
    bin_path = osp.join(cur_path, 'build', 'bin', 'bert_gemm')
    assert osp.exists(bin_path), f'{bin_path} not exists'
    return bin_path


def main(head_num: int = 16,
         size_per_head: int = 64,
         max_seq_len: int = 512,
         tensor_para_size: int = 1,
         max_batch_size: int = 256):
    for bsz in range(1, max_batch_size + 1):
        for seq_len in range(1, max_seq_len + 1):
            subprocess.call(
                f'{get_bert_gemm()} {bsz} {seq_len} {head_num} {size_per_head} 1 0 1 1',
                shell=True)


if __name__ == '__main__':
    import fire

    fire.Fire(main)