# org_result = org_result.tolist()
cm = confusion_matrix(test_gt, prompter_result)

# 将混淆矩阵存储到文件中
with open(f'{args.inter_var_dir}/{args.shot_numb}_shot_{args.dataset}_' + refined_template + '/cm.pickle', 'wb') as f:
    pickle.dump(cm, f)

plt.matshow(cm, cmap=plt.cm.Greens)
for i in range(len(cm)):
    for j in range(len(cm)):
        if round(cm[j, i] / sum(cm[j, :]), 2) == 0:
            continue
        plt.annotate(round(cm[j, i] / sum(cm[j, :]), 2), xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=5)
plt.ylabel('True Label')
plt.xlabel('predicted label')
plt.xticks(range(0, len(class_names)), labels=class_names, rotation='vertical')
plt.yticks(range(0, len(class_names)), labels=class_names)
plt.savefig(f'{args.inter_var_dir}/{args.shot_numb}_shot_{args.dataset}_' + refined_template + '/confusion_matrix.png', dpi=300)
plt.show()