# vroom
Learning Robotic Tasks from Video Observation

## convlstm usage
checkout alantest branch
`git checkout alantest`

Ensure checkpoints directory exists
`mkdir -p checkpoints`

Data should follow this example structure for train and test directories:
`./RoboTurk_videos/bins-Bread/test/demo_XXX_jointdata/frame_XXXX.npy`
`./RoboTurk_videos/bins-Bread/test/demo_demo_XXX/frame_XXXX.jpg`

Modify hyperparameters in trainer_lstm.py

Run trainer script
``python trainer_lstm.py --folder `realpath RoboTurk_videos/bins-Bread` --name lstm --save_best True --dataset roboturk``