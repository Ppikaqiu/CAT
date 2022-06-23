#!/bin/bash

# Copyright 2018-2021 Tsinghua University
# Author: Hongyu Xiang, Huahuan Zheng
# Apache 2.0.
# This script implements CTC-CRF training on WSJ dataset.
# It's updated to v2 by Huahuan Zheng in 2021, based on CAT branch v1 egs/wsj/run.sh

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
stage=0
stop_stage=8
g2p_lexicon=/home/qzy/CAT/egs/Indonesia/input-qzy/lexicon_id.txt

lang=indonesia
train_set=train_"$(echo "${lang}" | tr - _)"
train_dev=dev_"$(echo "${lang}" | tr - _)"
test_set=test_"$(echo "${lang}" | tr - _)"
recog_set="${train_dev} ${test_set}"

datadir=/home/qzy/CAT/egs/Indonesia/INDONESIA-v8
#datadir=/home/qzy/CAT/egs/Indonesia/INDONESIA-v8
. utils/parse_options.sh

NODE=$1
if [ ! $NODE ]; then
    NODE=0
fi

if [ $NODE == 0 ]; then
  if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Data Preparation and FST Construction"
    # Use the same datap prepatation script from Kaldi, create directory data/train, data/dev and data/test, 
    # create spk2utt, utt2dur and other files in each new directory.
    #local/mozilla_data_prep.sh $H $de $pure_word|| exit 1;
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl "${datadir}" ${part} data/"$(echo "${part}_${lang}" | tr - _)"
        cat data/"$(echo "${part}_${lang}" | tr - _)"/text | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | sed 's/\!//g' | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" > data/"$(echo "${part}_${lang}" | tr - _)"/text_fil
       mv data/"$(echo "${part}_${lang}" | tr - _)"/text_fil data/"$(echo "${part}_${lang}" | tr - _)"/text
       #        sed -i 's/mp3/wav/g' data/"$(echo "${part}_${lang}" | tr - _)"/wav.scp 
    done
   
    utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set}
    utils/filter_scp.pl --exclude data/${train_dev}/wav.scp data/${train_set}/wav.scp > data/${train_set}/temp_wav.scp
    utils/filter_scp.pl --exclude data/${test_set}/wav.scp data/${train_set}/temp_wav.scp > data/${train_set}/wav.scp
    utils/fix_data_dir.sh data/${train_set}
    
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3

    # # Construct the phoneme-based lexicon from the CMU dict
    local/mozilla_prepare_phn_dict.sh $g2p_lexicon || exit 1;
    ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
      data/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

    local/mozilla_train_lms.sh data/train_"$lang"/text data/dict_phn/lexicon.txt data/local/local_lm || exit 1;
    # Compile the language-model FST and the final decoding graph TLG.fst
    local/mozilla_format_local_lms.sh --lang-suffix "phn"
    local/mozilla_decode_graph.sh data/local/local_lm data/lang_phn data/lang_phn_test || exit 1;
   

  fi
  exit 0;
 

  if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "FBank Feature Generation"

    # Generate the fbank features; by default 40-dimensional fbanks on each frame
    fbankdir=fbank
    for set in train_"$lang" dev_"$lang"; do
      steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
      utils/fix_data_dir.sh data/$set || exit;
      steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
    done

    for set in test_"$lang"; do
      steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
      utils/fix_data_dir.sh data/$set || exit;
      steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
    done
  fi

  data_tr=data/train_"$lang"
  data_cv=data/dev_"$lang"  

  if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number || exit 1
    ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number || exit 1
    echo "convert text_number finished"

    # prepare denominator

    cat $data_tr/text_number | sort -k 2 | uniq -f 1 > $data_tr/unique_text_number
    mkdir -p data/den_meta
    #generate phone_lm.fst, a phone-based language model
    chain-est-phone-lm ark:$data_tr/unique_text_number data/den_meta/phone_lm.fst
    #generate the correct T.fst, called T_den.fst
    python3 ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
    #compose T_den.fst and phone_lm.fst into den_lm.fst
    fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
    echo "prepare denominator finished"
  
    # for label sequence l, log p(l) also appears in the numerator but behaves like an constant. So log p(l) is
    # pre-calculated based on the denominator n-gram LM and saved, and then applied in training.
    path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight || exit 1
    path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight || exit 1
    echo "prepare weight finished"
  fi

  if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    mkdir -p data/all_ark
    copy-feats "$feats_tr" "ark,scp:data/all_ark/tr.ark,data/all_ark/tr.scp" || exit 1
    copy-feats "$feats_cv" "ark,scp:data/all_ark/cv.ark,data/all_ark/cv.scp" || exit 1

    ark_dir=data/all_ark

    mkdir -p data/pickle
    python ctc-crf/convert_to.py -f=pickle data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
    python ctc-crf/convert_to.py -f=pickle data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
  fi

  data_test=data/test_"$lang"

  if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_test/utt2spk scp:$data_test/cmvn.scp scp:$data_test/feats.scp ark:- \
        | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    mkdir -p data/test_data
    copy-feats "$feats_test" "ark,scp:data/test_data/test.ark,data/test_data/test.scp"
  fi
fi

PARENTDIR='.'
dir="exp/blstm_earlyStop/"
DATAPATH=$PARENTDIR/data/

if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  unset CUDA_VISIBLE_DEVICES

  if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
    echo ""
    tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
  elif [ $NODE == 0 ]; then
    echo ""
    echo "'$dir/scripts.tar.gz' already exists."
    echo "If you want to update it, please manually rm it then re-run this script."
  fi

  # uncomment the following line if you want to use specified GPUs
  # CUDA_VISIBLE_DEVICES="0"                    \
  python3 ctc-crf/train.py --seed=0             \
    --world-size 1 --rank $NODE                 \
    --resume=$dir/ckpt/checkpoint.pt            \
    --batch_size=32                             \
    --dir=$dir                                  \
    --config=$dir/config.json                   \
    --data=$DATAPATH                            \
    || exit 1
fi

if [ $NODE -ne 0 ]; then
  exit 0
fi

nj=20
if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  for set in test; do
    ark_dir=$dir/logits/${set}
    mkdir -p $ark_dir
    CUDA_VISIBLE_DEVICES=0                          \
    python3 ctc-crf/calculate_logits.py             \
      --resume=exp/VGG/ckpt/bestckpt.pt                   \
      --config=$dir/config.json                     \
      --nj=$nj                                      \
      --input_scp=data/test_data/${set}.scp         \
      --output_dir=$ark_dir                         \
      || exit 1
 done
fi


if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  nj=20
  for set in test; do
     mkdir -p $dir/decode_${set}
     ln -s $(readlink -f $dir/logits/${set}) $dir/decode_${set}/logits
     ctc-crf/decode.sh --stage 1 \
     --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
     data/lang_phn_test data/test_id data/test_data/${set}.scp $dir/decode_${set}
  done
fi

#grep WER $dir/decode_test/cer_* | utils/best_wer.sh
grep WER $dir/decode_test/wer_* | utils/best_wer.sh



if [ $stage -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  for set in test; do
     for lmtype in bd_tg bd_tgpr; do
        # reuse logits
        mkdir -p $dir/decode_${set}_${lmtype}
        ln -s $(readlink -f $dir/logits/${set}) $dir/decode_${set}_${lmtype}/logits
        ctc-crf/decode.sh --stage 1 \
          --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
          data/lang_phn_test_${lmtype} data/${set}_pl data/test_data/${set}.scp $dir/decode_${set}_${lmtype}
     done
  done


  for set in test; do
    mkdir -p $dir/decode_${set}_bd_fgconst
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_phn_test_bd_{tgpr,fgconst} data/${set} $dir/decode_${set}_bd_{tgpr,fgconst} || exit 1;
    mkdir -p $dir/decode_${set}_tg
    steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 data/lang_phn_test_{tgpr,tg} data/test_${set} $dir/decode_${set}_{tgpr,tg} || exit 1;
  done
fi

