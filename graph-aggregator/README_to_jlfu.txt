1) Install packages
- Python>=3.6
- [PyTorch 0.4.0]

```bash
conda create -n graphie_env python=3.6
conda activate graphie_env
pip install -r requirements.txt
# install pytorch 0.4.0 in your own way
```

2) glove文件为glove.6B.100d，放在./data/pretr/emb/文件夹下

3）运行preprocess.py，将conll03等数据集处理为程序所需要格式的数据集，并将其放在data/dset/03co/文件夹下（目前文件夹中已有处理好的conll03数据集）

4）Run
python examples/multi_runs_conll.py --gpu_id 0 


5）结果文件存在./data/run/中，模型存在./data/model/中。


