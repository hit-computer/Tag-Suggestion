# Tag-Suggestion
这部分实验模型代码参考了[hed-dlg-truncated](https://github.com/julianser/hed-dlg-truncated)。不过他们的代码非常复杂，实现的功能也很多，我最终所借鉴的部分其实并不多。由于刚开始学习Theano时就接触到这份代码，当时也是一边参考theano官方文档一边研究这份代码的，因而最终自己的模型代码中借鉴一些他们已有的功能模块。


## 运行说明
在命令行中输入（使用GPU进行训练时）
    
    THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python train.py --prototype prototype_zhifu
    
如果没有GPU，则输入

    THEANO_FLAGS='mode=FAST_RUN,device=cpu,floatX=float32' python train.py --prototype prototype_zhifu
