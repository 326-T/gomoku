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

### 行動価値関数でターゲットネットワークなし

| Experience replay | fixed target q-network | reward clipping | 行動価値関数 | 価値関数 |
| ---- | ---- | ---- | ---- | ---- |
| o | x | o | o | x |

結果は`data/fn_q`に格納した。学習が全く安定しなかった。
`先行`
![image](https://user-images.githubusercontent.com/32381339/147847249-d32b0eb8-a5af-4652-a790-dbd64990c34f.png)
`後攻`
![image](https://user-images.githubusercontent.com/32381339/147847268-3aee0db4-7e8f-47a3-8db3-ec6fc9ce7eca.png)


