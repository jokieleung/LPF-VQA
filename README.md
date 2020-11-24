# LPF: A Language-Prior Feedback Objective Function for Out-of-distribution Generalization in Visual Question Answering

This is the code for the **NAACL 2021** submission.

All the data pre-process and projects' setup please refer to [project_setup.md](./project_setup.md) written by RUBi. Many thanks for their efforts.

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

## About Open Source

We will open-source the codebase upon acceptance.