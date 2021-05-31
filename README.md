# LPF: A Language-Prior Feedback Objective Function for De-biased Visual Question Answering

This is the code for the **SIGIR 2021** paper.

All the data pre-process and projects' setup please refer to [project_setup.md](./project_setup.md) written by [RUBi](https://github.com/cdancette/rubi.bootstrap.pytorch). Many thanks for their efforts.

The implementation of our model and the LPF objective function is in the folder:

`rubi/models/networks/LPF.py` and `rubi/models/criterions/lpf_criterion.py`.

## LPF works well on different VQA architecture

In this codebase, we implement LPF on 3 different VQA architecture: **UpDn**, **BAN** and **S-MRL**, which is in 

`rubi/models/networks/updn.py`, `rubi/models/networks/ban.py` and `rubi/models/networks/baseline_net.py`.

## Run the code

To run the code of LPF's training on the **VQA-CP v2**, please follow the script bellowed:

```
python -m bootstrap.run -o rubi/options/vqacp2/[model_name].yaml
```

**Note:** the `[mode_name]`can be `[lpf]`, `[lpf_ban]`, `[lpf_updn]`.

## Updates
This is the very begining version of [LPF-VQA](https://github.com/jokieleung/LPF-VQA). We will detail the README and code after several DDLs, thanks for your patience.

## Citation

If you find this paper helps your research, please kindly consider citing our paper in your publications.

```BibTeX
@inproceedings{liang2021lpf,
  title={LPF: A Language-Prior Feedback Objective Function for De-biased Visual Question Answering},
  author={Liang, Zujie and Hu, Haifeng and Zhu, Jiaying},
  booktitle={Proceedings of the 44th International Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2021}
}
```



