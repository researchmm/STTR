## Install

```
conda install -c pytorch pytorch torchvision
conda install cython scipy
```

## Test

Download pre-trained model here

https://drive.google.com/file/d/1OsRljnuWBb9BQzFJnidkhwuPKOt6YYx9/view?usp=sharing

Put it under ```./checkpoint_model```

Please change the images you want to use in the following folders:

```
inputs/content: a folder to store all content images
inputs/style: a folder to store all style images
```
Then run 

```
python demo_sttr_image.py
```

You could see results in ```outputs/test_outputs/0005```