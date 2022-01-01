# 強化学習で五目並べやオセロをやります

## 実装済み
- [x] 五目並べ
- [ ] オセロ
- [x] 対戦機能
- [x] DQN
  - [x] Experience replay
  - [x] fixed target q-network
  - [x] reward clipping
  - [ ] 行動価値関数
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
- ブラウザで以下にアクセスする。    
  http://localhost:5000/
  
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

結果は`data/fa`に格納した。
    
### 価値関数でターゲットネットワークあり

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | o | o | x | o |

結果は`data/dnn`に格納した。  
`人が先攻/人が後攻`  

<img src="https://user-images.githubusercontent.com/32381339/147853998-98dbc2fa-7a12-45ac-b6ec-8b18ccbc70a4.png" width="400px"><img src="https://user-images.githubusercontent.com/32381339/147854034-d1ee4155-d8e6-44fe-8e3f-238f15d771a7.png" width="400px">

### 行動価値関数でターゲットネットワークなし

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | x | o | o | x |

結果は`data/fn_q`に格納した。学習が全く安定しなかった。どちらも人の勝ち。  
`人が先攻/人が後攻`

<img src="https://user-images.githubusercontent.com/32381339/147847268-3aee0db4-7e8f-47a3-8db3-ec6fc9ce7eca.png" width="400px"><img src="https://user-images.githubusercontent.com/32381339/147847523-26bb7aa3-e272-4e32-9238-e5619d16ab00.png" width="400px">


