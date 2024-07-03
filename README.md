# train.py 실행 명령어

```bash
python train.py --feature_dim 12 --real real --fake fake --batch_size 16 --epochs 100 --model lstm --learning_rate 0.0001 --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4'
```
- feature_dim : 뽑아내는 feature와 mfcc 계수의 개수(차원)
- real : real 데이터 있는 폴더(이 안에 wav, feature_{feature_dim} 폴더 있음)
- fake : real 데이터 있는 폴더(이 안에 wav, feature_{feature_dim} 폴더 있음)
- model : 현재는 lstm, cnn중에 선택
- mfcc_feature_idx : 뽑아내는 행 번호(mfcc)
- evs_feature_idx : 뽑아내는 행 번호(evs)
    - 'all'의 경우 전부 다
    - 'none'의 경우 하나도 안뽑음

# 주의사항(수시로 업데이트됨)
- (2023/07/03) 조기종료, dropout 적용됨. feature_{feature_dim} 폴더 없을때 feautre_12에서 데이터 뽑고 zero padding 하게 수정함.
- (2024/07/02) mfcc는 pt파일로 저장하지 말고 그냥 음원에서 뽑자. 40차 이상이 넘어가니까 죽는다. 
- (2024/07/01) 현재 서버의 pt 파일들은 모두 정규화된 12차 데이터이다
- 현재 코드에서 mfcc_feature_idx를 'all'로 두면 고정된 filter back 수인 50이 모두 나옴
    - 우선은 수정하지 않음(추후에 수정 방향 생각)
- (2024/06/30) Dataset.py에서, feature_dim이 클 경우 pt파일로 저장하는 부분 주석처리 해야함
- model을 cnn으로 설정했을 경우, 'mfcc_feature_idx의 크기 + evs_feature_idx의 크기'가 8 이상이어야 함
- 예를 들어 '--model cnn --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4'' 로 설정하면 실행 안됨
- lstm은 상관 없음

- 현재 AudioDeepFakeDetectionNew 리포지토리에는 데이터가 있는 폴더는 있지 않다

- 현재 작업 폴더에 features_labels_mfcc_{feature_dim}.pt, features_labels_evs_{feature_dim}.pt 있으면 미리 저장된 데이터 불러온다
- 서버(선배님 컴퓨터)에서 현재 폴더의 pt 파일은 real은 LJ, fake는 LJ melgan 저장되어있다
- 학습할 때마다 features_and_labels 폴더에 학습 데이터를 pt 파일로 저장한다(추후 학습을 위해)
- 단, 명령 인수와 데이터 형태가 일치하는지 잘 확인해야한다!!

# inference.py 실행 명렁어
```bash
 python inference.py --model model_weights.pt --model_architecture lstm --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4' --feature_dim 12 --input_dir fake
```