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

## オセロ
1エピソードが長すぎて学習に時間がかかる。

