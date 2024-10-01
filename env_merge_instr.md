1. install.sh
2. update make files nodes and derivatives(brand-nsp/brand-emory)
3. compile =>make -j
4. change LM_ENV in config.simT16
5. to use pretrained rnn => copy 2024-09-05/RawData/Models/gru_decoder/rnn_model_0 use the same path change date to current date
6. in the "new" folder change args.yaml=>date in the op dir => required for online model
7. delete load_dir

9. open_vocab_closed_loop_training_rnn_000.txt => change all dates for RNN path to curr date
10. run /home/pdeevi/Projects/emory-cart/graphs/t16/speech/diagnostic.yaml first => gen blockmean files
11. copy layers to online_trainer_config from args.yaml?
12. When doing all sentence task testing on the simulator you have to turn off the analogue timestamp reading. It's a flag in sentencetask commandline
13. Then you have to turn off audio in the gru preprocess derivative
