# DA233X-Public
Public code for Master's thesis project in machine learning, DA233X at KTH, fall of 2018.

The data preprocessing is found in the "Data" folder, along with the raw datasets as referenced in Appendix A of the thesis.

In the "Code" folder, there are two sets of models which were used for the model comparison in the study. There is the CoT code from the paper [CoT: Cooperative Training for Generative Modeling](https://arxiv.org/abs/1804.03782) by Sidi Lu et al., which is taken from the [official repository](https://github.com/desire2020/CoT). Further, a SeqGAN model and an MLE model was implemented via the Texygen project as described in the SIGIR 2018 paper: [Texygen: A Benchmarking Platform for Text Generation Models](https://arxiv.org/abs/1802.01886) by Yaoming Zhu et al., with the code attained from the [official repository](https://github.com/geek-ai/Texygen). Full references are available in the report.

The computations were performed on the Hebbe cluster at Chalmer’s Centre for Computational Science and Engineering (C3SE) provided by the Swedish National Infrastructure for Computing (SNIC).

The code for the scoring and analysis of the generated text sets were created by the author, with a contribution from the Stack Overflow user rvinas' sample code for a [Keras Language Model perplexity](https://stackoverflow.com/questions/51123481/how-to-build-a-language-model-using-lstm-that-assigns-probability-of-occurence-f/51126064). All results, and the code, can be found in the "Results" folder, along with the results from the human evaluations.

For all code that has been created by the other authors mentioned above, modifications have been done to some files. This has not been done to change the algorithms, but to fit the code to the problem.