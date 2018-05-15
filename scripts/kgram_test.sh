#! /bin/sh
  
TRAIN_PATH=../k-grammar/grammar_std.train.full.tsv
DEV_PATH=../k-grammar/grammar_std.tst.full.tsv
TEST1=../k-grammar/grammar_repeat.tst.full.tsv
TEST2=../k-grammar/grammar_short.tst.full.tsv
TEST3=../k-grammar/grammar_long.tst.full.tsv

#MACHINE=[PATH-TO-MACHINE]
#EXPT_DIR=[PATH-TO-OUTPUT]
EMB_SIZE=64
H_SIZE=64
DROPOUT_ENCODER=0
DROPOUT_DECODER=0
CELL='gru'
EPOCH=200
PRINT_EVERY=10
SAVE_EVERY=10
ATTN='pre-rnn'
ATTN_METHOD='hard'
BATCH_SIZE=128

echo 'BASELINE'
python ${MACHINE}/train_model.py --train $TRAIN_PATH --dev $TEST1 --monitor $DEV_PATH  $TEST1 $TEST2 $TEST3  --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention $ATTN --epoch $EPOCH --save_every $SAVE_EVERY --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER --use_input_eos --teacher_forcing_ratio 0 

echo 'HARD ATTENTION'
python ${MACHINE}/train_model.py --train $TRAIN_PATH --dev $TEST3 --monitor $DEV_PATH  $TEST1 $TEST2 $TEST3  --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention $ATTN --epoch $EPOCH --save_every $SAVE_EVERY --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER --use_input_eos --teacher_forcing_ratio 0  --full_focus

echo 'SOFT ATTENTION'
python ${MACHINE}/train_model.py --train $TRAIN_PATH --dev $TEST3 --monitor $DEV_PATH  $TEST1 $TEST2 $TEST3  --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention $ATTN --epoch $EPOCH --save_every $SAVE_EVERY --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER --use_input_eos --teacher_forcing_ratio 0  --full_focus --use_attention_loss 







