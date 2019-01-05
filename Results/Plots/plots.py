import numpy as np
import matplotlib.pyplot as plt

# Data
b2 = [0.015961834482920698, 0.09197651663405088, 0.1608548603417109, 0.18160664578983995]
b3 = [0.0014769607731923678, 0.0089918526859641, 0.036746807162626294, 0.034913012674171455]
sb2 = [0.09769606331585981, 0.5301551288557595, 0.6577722522760012, 0.8056450353853819]
sb3 = [0.003777777777777777, 0.07301129424207743, 0.1907748679620704, 0.37056301654775387]

#u2g = [425515, 110887, 68596, 45984]
u2g = [94.20, 56.23, 44.97, 31.15]
#u3g = [440770, 181164, 119956, 101624]
u3g = [99.79, 96.76, 84.14, 73.80]

#rlm25 = [18.86014728401598, 19.41281130526293, 18.83161816499856, 20.472612770763433]
lm = [16.253407752441394, 9.77955082578344, 9.151639712687041, 8.116513430204567]
rlm = [8.440239682049258, 8.76460597810929, 8.83256385820577, 9.339328697832268]

sentlen_score = [43.78125, 38.56043956, 24.69863014, 14.40540541, 13.38333333]
colors_scores = np.array([[172,203,255], [146,187,255], [120,170,255], [100,158,255], [65,136,255]])

sentlen_model = [46.24, 20.75, 16.28, 15.78]

colors = np.array([[166, 206, 227], [31, 120, 180], [178, 223, 138], [51,160, 44]])
markers = ['o', '^', 's', 'D']
labels = ['CoT 130', 'CoT 50 + 80', 'MLE', 'SeqGAN']
'''
# 2-grams scatter
for i in range(4):
    plt.scatter(b2[i], sb2[i], c=colors[i]/255.0, marker=markers[i], label=labels[i], s=150)

plt.xlabel('BLEU2')
plt.ylabel('Self-BLEU2')
plt.title('2-gram BLEU scores')

axes = plt.gca()
axes.set_xlim([0,0.25])
axes.set_ylim([0,0.9])

plt.legend(loc='upper left')
plt.legend(fontsize=12)
plt.show()

# 3-grams scatter
for i in range(4):
    plt.scatter(b3[i], sb3[i], c=colors[i]/255.0, marker=markers[i], label=labels[i], s=150)

plt.xlabel('BLEU3')
plt.ylabel('Self-BLEU3')
plt.title('3-gram BLEU scores')

axes = plt.gca()
axes.set_xlim([0,0.05])
axes.set_ylim([0,0.4])

plt.legend(loc='upper left')
plt.legend(fontsize=12)
plt.show()

# bar charts
for i in range(4):
    plt.bar(i+1, u2g[i], color=colors[i]/255.0)
plt.xticks([1,2,3,4],labels)
plt.title('Unique 2-grams [%]')
plt.show()

for i in range(4):
    plt.bar(i+1, u3g[i], color=colors[i]/255.0)
plt.xticks([1,2,3,4],labels)
plt.title('Unique 3-grams [%]')
plt.show()

# LM scatter
for i in range(4):
    plt.scatter(lm[i], rlm[i], c=colors[i]/255.0, marker=markers[i], label=labels[i], s=150)

plt.xlabel('LM score')
plt.ylabel('Reverse LM score')
plt.title('Language Model scores')

axes = plt.gca()
axes.set_xlim([7,17])
axes.set_ylim([8,9.5])

plt.legend(loc='upper right')
plt.legend(fontsize=12)
plt.show()

# Sentence length per score bar
for i in range(5):
    plt.bar(i+1, sentlen_score[i], color=colors_scores[i]/255.0)
plt.xticks([1,2,3,4,5])
plt.xlabel('Human judgment score')
plt.ylabel('Average sentence length')
plt.title('Average sentence length per score')
plt.show()
'''
# Sentence length per model bar
for i in range(4):
    plt.bar(i+1, sentlen_model[i], color=colors[i]/255.0)
plt.xticks([1,2,3,4], labels)
plt.xlabel('Human judgment score')
plt.ylabel('Average sentence length')
plt.title('Average sentence length per model')
plt.show()