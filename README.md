# 強化学習で五目並べやオセロをやります

## 実装済み
- [x] 五目並べ
- [x] オセロ
- [x] 対戦機能
- [x] DQN
  - [x] Experience replay
  - [x] fixed target q-network
  - [x] reward clipping
  - [x] 行動価値関数
- [ ] DDQN
- [ ] APE_X


## 学習
- 以下を実行
  ```
  python trainer.py
  ```

## 対戦
- 以下を実行
  ```
  python player_match.py
  ```
- 五目並べならブラウザで以下にアクセスする。    
  http://localhost:5000/
 
- オセロなら以下にアクセスする。  
  http://localhost:5000/othello

<img src="https://user-images.githubusercontent.com/32381339/147877437-6109c1db-a38f-47d4-9c7c-0d114b415a93.png" width=400px>

  
## 三目並べ
- 先攻後攻はランダム
- 50000エピソード学習
- 一世代前のモデルと対戦させ学習
- 10世代分
- 最後に人と対戦
- 報酬のクリッピング（勝ち:1, 負け:-1, 引き分け:-0.3）
### 価値関数でターゲットネットワークなし    

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | x | o | x | o |

結果は`data/gomoku/fn_v`に格納した。どちらも人の勝利  
`人が先攻/人が後攻`  

<img src="https://user-images.githubusercontent.com/32381339/147854179-58a7c224-3464-42fb-8f94-90a81b31d260.png" width="400px"><img src="https://user-images.githubusercontent.com/32381339/147854138-31c045ee-c403-42dc-a868-944ed4a05de8.png" width="400px">


### 価値関数でターゲットネットワークあり

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | o | o | x | o |

結果は`data/gomoku/dnn_v`に格納した。どちらも人の勝利。  
`人が先攻/人が後攻`  

<img src="https://user-images.githubusercontent.com/32381339/147853998-98dbc2fa-7a12-45ac-b6ec-8b18ccbc70a4.png" width="400px"><img src="https://user-images.githubusercontent.com/32381339/147854034-d1ee4155-d8e6-44fe-8e3f-238f15d771a7.png" width="400px">

### 行動価値関数でターゲットネットワークなし

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | x | o | o | x |

結果は`data/gomoku/fn`に格納した。学習が全く安定しなかった。どちらも人の勝利。  
`人が先攻/人が後攻`

<img src="https://user-images.githubusercontent.com/32381339/147847268-3aee0db4-7e8f-47a3-8db3-ec6fc9ce7eca.png" width="400px"><img src="https://user-images.githubusercontent.com/32381339/147847523-26bb7aa3-e272-4e32-9238-e5619d16ab00.png" width="400px">

### 行動価値関数でターゲットネットワークあり

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | o | o | o | x |

結果は`data/gomoku/dnn`に格納した。学習が全く安定しなかった。どちらも人の勝利。  
`人が先攻/人が後攻`  

<img src="https://user-images.githubusercontent.com/32381339/147854289-3f6972c2-53dc-475e-b8d0-561bc50d7b61.png" width="400px"><img src="https://user-images.githubusercontent.com/32381339/147854269-a1766141-a281-4843-8422-23136fd3976c.png" width="400px">

## 考察
### 行動価値関数と価値関数について  
価値関数を用いた方が学習が安定していた。  
次の状態がもっとも高くなる行動以外の行動についても誤差が計算されるので。つまり他の行動後の状態の価値が極力変化しないように学習されていたと考えられる。  
- 上側：行動価値関数
  - 左：ロス
  - 右：報酬
- 下側：価値関数
  - 左：ロス
  - 右：報酬

![0_reward_loss](https://user-images.githubusercontent.com/32381339/148748701-e37a4824-4ec6-4d24-8661-5ffdf7024ab4.png)
![0_reward_loss](https://user-images.githubusercontent.com/32381339/148748352-6fe8a439-cf29-4a66-ac61-cd6d52633163.png)

### マルチエージェント強化学習
より強いエージェントを得るために学習済みのモデル同士を対戦させた
- 対戦相手はエージェントの初期状態と同じモデル
- 対戦相手は学習しない
- 50000エピソードの後、対戦相手のモデルを更新

結果としては学習が安定しなかった。理由としては以下だと思っている。
- 代を重ねるごとに破滅的忘却によって以前対戦したエージェントを忘れている。
- 各世代ごとに固有の行動(例えば、斜めを優先するとか)が現れることがあり、その行動のみを対策するようになってしまっている。

また行動価値関数を用いた方が、上記の傾向が強かった。

## オセロ
1エピソードが長すぎて学習に時間がかかる。

### 行動価値関数でターゲットネットワークあり

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | o | o | o | x |

結果は`data/othello/dnn`に格納した。
エージェントが先攻時でも人には勝てなかった。

<img src="https://user-images.githubusercontent.com/32381339/148741854-a7083d96-15ea-4603-a70f-21d57c292039.png" width="400px">


学習の様子
- 左：ロス
- 右：報酬

![0_reward_loss](https://user-images.githubusercontent.com/32381339/148749970-ef1ef529-aacc-4997-b68a-b2ee8f0dbe35.png)

そもそも学習できてない。原因は以下が考えられる。
- 状態が広すぎる
- エピソードの終わり付近以外は報酬は0
- 報酬がスパース過ぎること

角や辺をとったときに中間報酬を与えることも有効かもしれない。再度学習させるだけの時間がないが。

