# MCM_ICM__2024
## 2024 MCM_E

##### Thanks my Great friends for their help!

### 问题1：Topsis评分+定价策略优化

##### Topsis评分：见程序：topsis_score.ipynb

- 数据预处理：整合美国各州各项数据
- 数据清洗：去除缺失值、异常值
- 数据标准化：将数据转换为0-1之间的小数
- 建立权重矩阵：根据各指标的权重，计算出各指标的权重向量
- 建立判断矩阵：根据各指标的关联程度，计算出各指标的关联矩阵
- 计算得分
- 计算得分排名



##### 定价策略优化：详情见程序：maxmize.ipynb

- 确定决策变量P，Q，X
- 确定约束条件：详情见程序
- 确定目标函数：Min Z=P+Q+X
- 使用Python中的gurobipy库求解

##### 风险系数绘图：详情见AISADATA.ipynb

- 数据预处理：整合美国各州各项数据
- 数据清洗：去除缺失值、异常值
- 调用plotly库绘制图形


### 问题3：AHP层次分析法辅助分析地标价值

##### 程序 recoginize.ipynb# MCM_ICM_2024_E


