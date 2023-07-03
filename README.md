# level2_cv_semanticsegmentation-cv-09
# Segmentation Wrap-Up Report

1. **프로젝트 개요**
    
    Hand bone x-ray 객체가 담긴 이미지에 대해 각 픽셀 좌표에 따른 bone class를 예측한다.
    

1. **프로젝트 팀 구성 및 역할**
    
    김지범: PSPNet 모델 실험, mmsegmentation, Optuna Hyperparameter tuning
    양경훈: DeepLabV3+ 모델 실험, BentoML 기반 모델 deploy 연구
    이경봉: Unet, segformer 모델 실험, Visualization
    이윤석: 베이스라인 구축, HRNet 모델 실험
    정현석: Unet variation 모델 실험, FrontEnd 모델 deploy 연구
    

1. **프로젝트 수행 절차 및 방법**
    1. **타임라인**
        
        ✓ 협업 convention 정의
        ✓ Pytorch 베이스라인 구현
        ✓ 각자 전담 model 에 대해 실험, 논문 소개 과정 진행
        ✓ 높은 리더보드 점수의 model output 을 모아 Ensemble
        
    2. **협업 문화**
        
        ✓ 피어 세션 때 각자 진행상황 및 이후 계획 공유
        ✓ 진행된 것이 있으면 실시간 Slack 에 공유 및 피드백
        ✓ Git Commit Convention 채택
        ✓ Issue 생성 -> branch 생성 -> Pull Request 형식의 코드 협업 Convention 채택
        
    3. **EDA**
        - 각 클래스별 비율
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6967eb60-1207-409a-95f5-a7e26860c837/Untitled.png)
            
            일반적으로 팔 뼈가 대부분을 차지한다. 손가락 마디 뼈들은 겹치는 부분이 크게 많지
            않아, 대체로 Segmentation 이 잘 진행될 거라 예상했다. 대체로 손목 부분의 여러
            뼈가 겹쳐 있는 부분에서 loss 가 크게 발생할 거라 생각했다. 해당 부분을 잘
            Segmentation 하는 것이 이번 대회의 키포인트라 생각했다.
            
        - Input Image 분석
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1850d3f-ce89-4b46-97ff-e14c92d66662/Untitled.png)
            
            크게는 2 가지 각도의 사진으로 이루어져 있다. Testset 도 이러한 분포를 가질거라 생각하고, Data Augmentation 기법 중 RandomRotate 를 적은 범위로 적용시키자 결정했다. 큰 범위를 적용하지 않은 이유는 손목이 꺾인 이미지에 대해 더 큰 Rotation 이 적용될 수 있음을 고려했다. 또한 각 사람마다 오른손, 왼손 이미지가 존재하는데, 때문에 Horizontalflip 기법을
            적용시킬지 조원들과 조사를 해보았을 때, 사람이 사용하는 빈도나 용도에 따라 오른손 왼손 뼈 형태가 다를 수 있다는 도메인 지식을 얻은 후, Horizontalflip 기법을 채택했다.
            
        - 이상치 데이터셋 제거
            
            의학 도메인 지식은 없지만 반지, 기능을 알 수 없는 조형물 등 X-ray 이미지에서
            일반적이지 않은 이미지들이 라벨을 예측하는데 방해가 될 거라 판단해 해당
            사진들을 모두 데이터셋에서 제외했다
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0b5ad748-ad19-46c8-b3c2-73496263e440/Untitled.png)
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/413074ab-9dda-45fd-9d8d-fb967e1aa895/Untitled.png)
            
            위의 이미지와 같이 validation dice 점수가 급격히 떨어지는 경우가 있었는데, 해당 부분이 위와 같은 특정 이미지 때문이 아닐까 판단했고, 해당 데이터를 제거하고 나서 해결됨
            
    
    d. **Data Augmentation**
    
    EDA 에서 분석한 대로, 우리의 데이터셋 이미지 형태 분포에 따라 Horizontal flip 과
    Random Rotate 를 적용시키면 좋을 것이라 판단했고, 각 Augmentation 기법의
    효용성 증명 실험을 진행했다.
    
    - Horizontal flip
    horizontal flip 을 기본 베이스라인과 비교해 실험해봤다. Horizontal flip 은 dice 값이
    약 0.005 상승했다.
    - Random Rotate
    Random Rotate 를 기본 베이스라인과 비교해 실험해 보았다. Random Rotate 를
    적용한 모델의 output 의 dice 값이 약 0.003 상승했다.
    - Random Rotate + Horizontal flip
    Augmentation 이 둘 다 적용된 모델에 대해 실험해 보았다. 베이스라인 코드에 비해
    output 의 dice 값이 0.004 가량 상승했다. 값 자체는 Horizontal flip 단독 적용
    모델보다 떨어지지만, 그 양이 크지 않고, 모델의 일반화를 위해 해당 Augmentation
    기법을 통제 변인으로 지정했다.
    - Random Crop
    해당 기법의 경우 베이스라인에 비해 dice 값이 큰 폭(0.12)으로 떨어져 채택하지
    않기로 결정했다.
    - Augmentation 결론
    의학 도메인 지식은 부족하지만, 뼈라는 target 의 특성은 대체로 안정적인 형태로
    쉽게 구부러지거나 부러지지 않는다. 또한 X-ray Image 의 Data Augmentation 기법에
    관한 reference 를 찾는 도중 “The Effectiveness of Image Augmentation in Deep
    Learning Networks for Detecting COVID-19: A Geometric Transformation
    Perspective”라는 저널에서 연구한 결과를 얻게 되었는데, 4 가지의 Augmentation
    군집으로 나누어 Augmentation 기법의 효과를 알아보고자 했는데 Augmentation 을
    적용시키지 않은 모델보다 성능이 떨어진다는 연구결과를 접했다. 따라서 결론적으로
    Random Rotate + Horizontal flip 을 대부분의 Augmentation 기법으로 채택했다.
    - CutMix, MixUp
    겹쳐진 뼈 부분에 대해 정확도가 낮은 부분을 개선하고자 하였다.
        
        -> Mixup, Cutmux 로 다양한 데이터를 만들어 실험.
        Mixup 을 사용한 오버랩 된 데이터를 통해 겹쳐진 뼈에 대한 학습을 해 더 좋은
        성능이 나온 것이라고 생각이 된다.
        Cutmix 를 통해 주변 뼈들에 대한 정보의 영향을 줄이고, target 에 대한 정보만을
        보도록 하여 겹쳐진 부분에 대한 정확도가 개선되었다.
        Base < cutmix < mixup 순으로 성능향상을 볼 수 있었다.
        
    
    e. **Model Select**
    
    각자 특정 모델에 대해 학습 후, Ensenble을 통해 성능 향상을 하는 방법으로 진행하였다.
    
    - PSPNET
        
        실험은 steplr scheduler 추가하기, backbone effiecient net 으로 바꿔보기, input size
        1024 로 바꿔보기, train set 중 이상한 사진 제거하기와 PSPNet 을 segmentation
        model pytorch 를 이용해서 tuning 해보기(encoder depth 늘려보기)를 해봤으나 val set
        기준 오른 것은 input 을 1024 로 바꾼 것 만이었다.
        steplr scheduler 추가 시 성능이 잘 안 나온 것은 0.007 정도의 차이인 것을 보아 큰
        의미는 없어 보인다.
        train 시 이상한 사진들을 제거한 실험이 성능이 오히려 감소한 이유는 데이터 셋이
        적다 보니 이상한 사진에서도 정상적인 부분들이 오히려 특징 추출했을 때 도움이
        되었던 것으로 보인다.
        encoder depth 를 늘려본 tuning 은 성능이 오히려 떨어졌다. 아마 그 이유는
        depth 를 늘렸던 것이 gloabal contextual information 을 더 잘 알아내도 작은 부분의
        특징들을 잘 학습하지 못해 성능이 안나오는 것으로 보인다.
        input 을 1024 로 바꾼 것은 역시 뼈의 특징들을 학습하는데 큰 도움이 되어 성능이
        val set 에서는 크게 올랐으나 왜 인지 실제 test set 에서는 성능이 안 좋은 것을 알
        수 있었습니다. train set 에 너무 overfitting 된 것으로 보인다
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55e33f00-3f86-4bc5-8fc7-c522d83ee115/Untitled.png)
        
    - Unet
        
        논문 제목에서도 볼 수 있듯, Biomedical Image 에 대한 Segmentation 논문. 해당
        논문의 벤치마킹 대회는 전자 현미경의 세포 관찰 이미지지만, 같은 medical Image 를
        target 으로 한다는 점에서 괜찮은 성능을 낼 것이라 판단했다.
        초기 Resnet34 Backbone 을 좀 더 파라미터가 많은 Resnet101, EfficientNet-b7 으로
        변경해 학습을 진행했을 때 모델 파라미터의 수 기준으로 높은 리더보드 점수를
        기록하는 점을 확인하고, EfficientNet-b7 으로 encoder model 을 고정했다.
        Unet 논문을 읽어보고 발표자료를 만들던 중, Unet 논문의 architecture 와
        SMP 라이브러리의 기본 Architecture 가 다르다는 점을 확인했다. 따라서 model 을
        직접 콘솔창에 출력해 본 결과
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf7e77ca-d07a-4ab3-9a6e-45e20226c129/Untitled.png)
        
        위와 같이 적은 체널에서 많은 체널로 Conv 연산이 일어나는 부분과, 너무 많은
        체널에서 적은 체널로 Conv 연산이 일어나는 부분을 확인했다. 이 부분에서 정보의
        손실이 크게 일어날 것이라 판단 했고, Decoder_channels 파라미터를 수정해 체널
        수가 적정 수준(1/2)으로 감쇄되도록 바꾸었다. 결과적으로 아래와 같은 유의미한
        결과를 얻을 수 있었다
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ee61482-13f9-41df-bcb3-195bfe717875/Untitled.png)
        
    - Unet++
        
        deep-supervision 을 적용한 모델과 적용하지 않은 모델 두 가지의
        UNet++(NestedUnet) 모델의 스크립트를 작성하여 모델 학습을 진행했습니다.
        deep-supervision 모델 스크립트 안에는 해당 기능을 구현하기 위한 if-else 분기문이
        존재하였고 이 분기문이 torch.jit.trace 방식으로 torchScript 변환하는 것을
        막고있었습니다. torch.jit.script 라는 대안이 있었지만 해당 방식으로의 변환이 경우에
        따라 제대로 변환되지 않는 이슈가 있었기 때문에 잘 알지 못하는 상황에서 섣불리
        사용하는것이 옳지 않다고 판단했습니다.
        두 버전의 모델의 성능을 비교해본 결과 0.002 의 성능차이밖에 나지 않았던 것을
        근거로 deep-supervision 을 사용하지 않는 모델을 사용하는것으로 결정하였고
        torch.jit.trace 방식으로 torchscript 로 추출하였습니다.
        
    - DeepLabV3+
        
        모델의 성능 향상을 위해 동일한 DeepLabV3+ 모델을 기반으로 백본 네트워크를
        변경해가며 실험을 진행하였습니다. 먼저 기존 베이스코드를 사용한 결과 모델의
        성능이 수렴하기까지 30 에포크 이상이 소요되었기에 네트워크의 파라미터 수가
        증가함에 따라 성능이 향상될 것이고, 그 속도도 빠를 것이라는 가설을
        설정하였습니다. 실험한 네트워크는 ResNet101, ResNet152, EfficientNet_B5,
        EfficientNet_B7, RegNetX_320 입니다. 파라미터의 수는 RegNetX_320, EfficientNet_B7,
        ResNet152, ResNet101, EfficientNet_B5 순으로 많았고, val/loss 와 val/dice 모두
        유사한 순서로 수렴하는 모습을 확인할 수 있었습니다. 성능의 최대치 역시 같은
        유사한 순서를 보였습니다
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81bd49dd-982c-414f-9fdf-b9b5660bd26a/Untitled.png)
        
        이후 이미지의 크기를 기존 설정인 (512, 512)에서 (1024, 1024)로 변경하여 실험을
        진행하였습니다. 실험에 앞서 이미지 크기의 변경은 성능 향상시키지만, 크기에
        비례해 4 배로 소요 시간이 증가할 것이라는 가설을 설정하였습니다. 실험에 사용한
        네트워크는 ResNet101, RegNetX_320 입니다. 실험 결과 ResNet101 에서는 수렴
        속도의 차이가 두드러졌지만, RegNetX_320 에서는 차이가 적었습니다. 그러나 실제
        제출 결과는 ResNet101 이 0.9495 에서 0.9683 로, RegNetX_320 이 0.9524 에서
        0.9680 으로 향상되었습니다. 이는 크기가 큰 이미지에서 일반화가 더 잘 이루어졌기
        때문에 나타난 결과라고 판단하였습니다. 또한 학습 시간 역시 기존 (512, 512)의
        경우에 ResNet101 과 RegNetx_320 이 각각 약 6 시간 20 분, 약 7 시간 20 분이
        소요되었지만, (1024, 1024)의 경우에 약 8 시간 30 분, 약 9 시간 10 분이 소요되어
        예상보다 적은 차이를 보였습니다. 이는 작은 이미지의 경우에 GPU 의 속도보다
        CPU 에서 GPU 로 데이터를 옮기는 속도가 떨어지기 때문에 나타난 현상이라고
        판단하였습니다.
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9f15be44-e15a-4c0e-9a2e-5e76aa56c4e2/Untitled.png)
        
        이후 위와 같은 결과를 바탕으로 GPU 자원을 가장 많이 사용할 수 있는, 다시 말해
        파라미터가 가장 많은 RegNetY_320 을 (1024, 1024) 크기의 이미지를 데이터로
        사용하여 0.9684 로 성능을 향상시켰습니다.
        
    - HRNet
        
        hrnet + ocr(object context representation) 코드 추가
        dataset 이미지가 2048x2048 의 큰 resolution 을 가지고 있고 또한 의학 이미지이므로
        전문적인 지식은 없어 정확한 판단은 아니지만 눈에 잘 보이지 않는 뼈의 뒤틀림
        같은 경우를 생각해봤을 때 약간의 픽셀 변화도 중요하다고 생각했습니다. 그리하여
        high-resolution 의 정보를 유지하는 모델 HRNet 을 선택
        또한 뼈의 종류들이 한 이미지당 하나의 object 만 있어 픽셀 별 classify 보다는
        object region 정보를 추출하는 ocr 구조를 선택해 object 별 classify 를 하도록 하였습니다.
        
    - Segformer
        
        초기 데이터셋을 살펴보면서 패치단위로 인코딩하는 transformer 모델이 5 개의
        손가락 뼈를 헷갈리지 않고 분류하는데 유리하지 않을까? 라는 추측을 했다.
        하지만 이미 FCN 베이스라인 코드부터 우려한 결과는 나오지 않았다(ex: 중지 마디를
        검지, 약지로 오 분류하는 상황)
        따라서 이전 대회에서 여러 특성이 다른 모델을 Ensemble 했을 때, 좋은 결과가
        나온다는 경험에 따라, 일정 이상 성능을 낼 수 있도록 Tuning 하고자 노력했고, 이때
        input Resolution 을 512 -> 1024 로 높였을 때, 모델의 성능이 비교적 많이 상승했다.
        사용한 모델들 중 제일 최신 모델로 알고 있지만, 해당 모델의 논문 내용을 자세하게
        읽어보지 못해 아쉬움이 남는다
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a82502c-17cb-4ce9-9654-a7aab1161c48/Untitled.png)
        
    
    f. **Loss**
    
    초기 베이스라인 코드에서는 BCELoss 를 사용했다. 이 때 문제점이 수렴속도가 현저히
    느렸다
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/efee6d7b-b375-4165-9a77-9eddc8d472e1/Untitled.png)
    
    위 그래프에서 수렴속도를 볼 수 있다. 따라서 Aistages 게시판에 올라온 DiceLoss +
    BCELoss CombineLoss 를 적용했고 위 그래프의 청록색 lineplot 과 같이 수렴속도가
    빨라졌다. 이 때문에 팀원들이 각자의 실험을 빠르게 적용해 볼 수 있었다.
    해당 부분에 대해 분석을 해보자면, 기본적으로 BCELoss 는 classification Loss 이다.
    따라서 픽셀 단위로 label 에 대한 loss 를 측정한다. 이것은 이미지의 모든 픽셀에서
    각각 loss 가 발생한다 할 수 있는데, 우리의 이미지의 경우 객체 못지않게 넓은
    범위의 배경이 존재한다. 따라서 이 부분이 학습 속도를 느리게 한 부분이지 않을까
    생각한다.
    이에 비해 DiceLoss 는 영역 간 겹쳐지는 정도를 Loss 로 측정한다. 따라서 배경과는
    상관없이 Prediction 과 GT 의 영역이 일치하는 정도에 따라 Loss 가 결정되므로,
    Segmentation 에는 DiceLoss 가 좀 더 적합한 Loss 로 볼 수 있고, 학습속도를 빠르게
    해주는 역할을 해주었다고 판단했다. DiceLoss 를 단독으로 썼을 때와 비교하는
    실험결과가 있었으면 좋았을 텐데, 해당 부분이 아쉽다
    
    g. **Visualization**
    
    - Wandb Visualization
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/508b7abb-52bb-4ffe-a1e6-d3d835324428/Untitled.png)
        
        위 2 개의 Visualization 을 wandb 에 로깅하도록 했다. 상단 막대그래프는 class 별
        Dice 값이고, 하단 이미지는 val image 중 segmentation 결과를 Visualization 한
        결과이다. 해당 결과를 wandb 에서 비교하며 모델의 학습 정도를 파악하고 class 별로
        잘 예측을 하지 못하는 뼈 번호도 알 수 있었다. (20, 26) 조금 아쉬움이 있는 부분은
        multilabel 문제라 아래에는 output 보다, loss heatmap 이 조금 더 어울렸을 듯하다
        
    
    h. **ETC**
    
    - Hyperparameter Tuning
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/827968bd-1d43-4e6d-bab2-8156a5444a24/Untitled.png)
        
        hyperparameter-tuning 을 위해 Optuna 를 사용해 봤다. 8 번의 트라이얼을 돌려서
        제일 좋게 나온 hyperparameter 대로 세팅을 했으나 train loss 는 낮게 잘 나왔으나
        validation 의 dice 값이 이상하게 나와 당황스러웠다. Wandb sweep 이나 Optuna 와
        같은 Hyperparameter search tool 을 사용하기 위해선 실험 초기부터 계획을
        체계적으로 수립해야 한다는 점을 깨달았다
        
    - Streamlit 활용
        1. Gt 와 predict 의 차이 plot
        Predict 와 GT 이미지 결과값의 차이를 보며 어떤 부분이 많이 틀리는 지 확인함.
        특히 20, 26 번의 다른 뼈와 겹치는 class 가 차이가 많이 발생하는 것을 확인
        2. 각 클래스 별 grad cam 확인 코드
        정확도가 낮은 클래스의 gradcam 을 확인하여 weight 가 어떤 부분에 중점을 두고
        있는 지 확인. GT 의 뼈 보다는 아래로 치우쳐지거나 다른 쪽으로 치우친 예측 경향을
        확인
        3. 각 클래스 별 losscam(loss heatmap) 확인 코드
        대부분의 데이터에서 뼈 가장자리 부분의 작은 픽셀들에서 loss 눈에 띄는 loss 를
        관찰할 수 있었고, 특히 겹쳐지는 부분에서 크고 선명한 losscam 을 확인할 수
        있었다.
    - Android Serving
        1. Onnx Opset 업그레이드
        이전 대회때와 비교하여 opset version 11 → opset version 15 으로 버전을 올렸으며
        성능 비교를 해본 결과 torch 모델과 변환된 onnx 모델의 출력차이가 Mismatched
        elements: 1 / 7602176 로 단 하나의 오차만 존재하도록 변환이 된 것을
        확인했습니다
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/90705aa6-bfc4-4973-8a21-f74e35d61f3b/Untitled.png)
            
        2. FrontEnd serving
            
            두 모델의 출력이 동일한 것을 검증하고 나서 Android serving 을 시도했지만 실제로
            모바일 기기로 이식했을 때 prediction 값이 많이 망가졌습니다.
            0.5 threshold 를 넘긴 값이 torch 환경에서는 459957 개, android 환경에서는
            1211 개로 성능이 거의 나오지 않았습니다. Threshold 를 크게 낮추어 visualize
            해보아도 어느정도 손가락 뼈 마디의 형상은 보였지만 정상적인 출력이 되지는
            않았습니다.
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8b76baa8-45b9-4ff2-a94f-819a2ef45b68/Untitled.png)
            
            파이썬 실행환경과 최대한 비슷하게 맞춰보기 위해 NDK OpenCV 라이브러리를 활용하는 방향으로 고민을 해보고 DNN 모델로 load 하여 출력을 뽑았지만 마지막에 시간이 부족하여 시각화까지는 하지 못하였습니다
            
    1. **프로젝트 결과**
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e7f76205-fd62-4752-8f28-0c8e9551b2e3/Untitled.png)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce3f838a-e121-4afb-968d-12c65c88e331/Untitled.png)
        
        public 0,9727 -> private 0.9733  최종 6 등
        
    2. 자체 평가 의견
        1. 잘했던 점
        - 여러 모델에 대해 각자 공부하고, 팀원들에게 발표하는 시간을 가진 점.
        - 총 3 주 기간 중 2 주만 투자했음에도 불구하고 높은 성적을 거둔 점.
        - 모델 구조를 파악하고 튜닝을 해본 점.
        - CombineLoss 채택 후 실험의 효율성을 가져온 점.
        
          b.  시도했으나 잘 되지 않은 것들
        
        - Soft Ensemble: 메모리 이슈를 해결하는 아이디어를 내지 못해 구현 못한 점.
        - Transformer model 채택: Segformer 와 mmseg 의 여러 모델들을 채택하지 못한 점
        
          c.  아쉬운 점
        
        - 비교적 최근 논문을 사용하지 못해본 점.
        - Serving 관점에서 생각을 하지 못한 점.
        - 도메인 지식이 부족한 점.
        
          d.  배운 점
        
        - Semantic Segmentation 여러 모델의 이점, 발전 이유