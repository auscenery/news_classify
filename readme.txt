cd news_classify

# 预测
python3 -m main.start_predict -mn best_validation-2400 -gf 0

# 启动分类后台服务
python3 -m main.get_label

# 训练 gf 1:gpu 0:cpu, 最好gpu训练
python3 -m main.start_train -gf 1