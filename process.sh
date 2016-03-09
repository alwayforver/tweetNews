time python dataSplitJesse122115.py 60 1 test1221/ test30.pickle
time python dataSplitJesse122115.py 39 1 test1221_3/ test30.pickle

cat /srv/data/jackyin/tweet17m.json |python toy.py "sanders vermont bernie independent senator burlington mr money senate class" 3 0


# 0123 has 1. wrong dbpedia parser (a. no leading space b. narrow definition of entity types only starting with 'dbpedia:') 2. did not address duplicates in news 3. pickle is using flat Xs and Terms 
########## tokenize news ###############
python tokenize_news.py /srv/data/jingjing/eknot/news/ ../data/news_tokenized/ 2014-11-18 2015-06-05
python tokenize_news.py /home/wtong8/NewsTwitter/news/ ../data/news_tokenized/ 2015-09-07 2016-01-23
    # data hacks
    cut -f11 2016-01-10.txt|awk '{print NF}'|histogram.py --percentage --max=1200 --min=0
# vectorize news; save to pickle
python vectorize_news.py ../data/news_tokenized/ 2016-01-07 2016-01-23 ../data/20160107_0123.pickle

time python vectorize_news.py ../data/news_tokenized_20160307/ 2016-01-07 2016-01-14 ../data/20160107_0114.pickle
"""
iterator started...
file processing: 2016-01-14.txt
final count = 835

done in (sec):  4.1397600174
n_samples, n_features:  (835, 14118)
iterator started...
file processing: 2016-01-14.txt
final count = 835

done in (sec):  1.9231338501
n_samples, n_features:  (835, 521)
iterator started...
file processing: 2016-01-14.txt
final count = 835

done in (sec):  2.11176991463
n_samples, n_features:  (835, 469)
iterator started...
file processing: 2016-01-14.txt
final count = 835

done in (sec):  2.13593101501
n_samples, n_features:  (835, 317)
iterator started...
file processing: 2016-01-14.txt
final count = 835

done in (sec):  4.33659911156
n_samples, n_features:  (835, 1307)
iterator started...
file processing: 2016-01-14.txt
final count = 835

done in (sec):  4.09358310699
n_samples, n_features:  (835, 4600)

real    0m22.618s
user    0m20.602s
sys     0m0.356s

"""
"""
[jwang112@dmserv4 scripts]$ python vectorize_news.py ../data/news_tokenized/ 2016-01-07 2016-01-23 ../data/20160107_0123.pickle
iterator started...
file processing: 2016-01-23.txt
final count = 1828

done in (sec):  9.19359302521
n_samples, n_features:  (1828, 21623)
iterator started...
file processing: 2016-01-23.txt
final count = 1828

done in (sec):  3.48802804947
n_samples, n_features:  (1828, 691)
iterator started...
file processing: 2016-01-23.txt
final count = 1828

done in (sec):  4.65621304512
n_samples, n_features:  (1828, 832)
iterator started...
file processing: 2016-01-23.txt
final count = 1828

done in (sec):  3.66906189919
n_samples, n_features:  (1828, 490)
iterator started...
file processing: 2016-01-23.txt
final count = 1828

done in (sec):  6.70147418976
n_samples, n_features:  (1828, 2013)
iterator started...
file processing: 2016-01-23.txt
final count = 1828

done in (sec):  6.81804895401
n_samples, n_features:  (1828, 4976)
"""
# new correct
"""
[jwang112@dmserv4 scripts]$ python vectorize_news.py ../data/news_tokenized_201
60131/ 2016-01-07 2016-01-23 ../data/20160107_0123.pickle
iterator started...
file processing: 2016-01-23.txt
final count = 1788

done in (sec):  29.0954179764
n_samples, n_features:  (1788, 21438)
iterator started...
file processing: 2016-01-23.txt
final count = 1788

done in (sec):  11.61774683
n_samples, n_features:  (1788, 710)
iterator started...
file processing: 2016-01-23.txt
final count = 1788

done in (sec):  11.8252542019
n_samples, n_features:  (1788, 786)
iterator started...
file processing: 2016-01-23.txt
final count = 1788

done in (sec):  8.9661128521
n_samples, n_features:  (1788, 469)
iterator started...
file processing: 2016-01-23.txt
final count = 1788

done in (sec):  19.6247091293
n_samples, n_features:  (1788, 1965)
iterator started...
file processing: 2016-01-23.txt
final count = 1788

done in (sec):  21.0310759544
n_samples, n_features:  (1788, 4848)
"""
# inits
time python eknot_init.py 30 /home/wtong8/NewsTwitter/tweets/   ../data/20160107_0123.pickle ../data/inits_20160107_0123_30.pickle > ../output/out20160107_0123_30inits.txt
# plsa
time python eknot.py ../data/20160107_0123.pickle ../data/inits_20160107_0123_30.pickle /home/wtong8/NewsTwitter/tweets/ ../output/plsa_20160107_0123_30.pickle> ../output/out20160107_0123_30.txt
# stats
# time python eknot_stats.py ../data/20160107_0123.pickle ../output/plsa_20160107_0123_40.pickle > ../output/stats_20160107_0123_40.txt  # old
time python eknot_stats.py ../data/20160107_0123.pickle ../output/plsa_20160107_0123_40.pickle null text> ../output/statsText_20160107_0123_40.txt
# sub
time python eknot_sub.py ../data/20160107_0123.pickle  ../output/plsa_20160107_0123_40.pickle null text 9 3 > ../output/.txt

# 0129
"""
[jwang112@dmserv4 scripts]$ python vectorize_news.py ../data/news_tokenized_20160131/ 2016-01-07 2016-01-29 ../data/20160107_0129.pickle
iterator started...
file processing: 2016-01-29.txt
final count = 2476

done in (sec):  43.5901391506
n_samples, n_features:  (2476, 25459)
iterator started...
file processing: 2016-01-29.txt
final count = 2476

done in (sec):  10.4302899837
n_samples, n_features:  (2476, 906)
iterator started...
file processing: 2016-01-29.txt
final count = 2476

done in (sec):  11.8986270428
n_samples, n_features:  (2476, 1009)
iterator started...
file processing: 2016-01-29.txt
final count = 2476

done in (sec):  13.6138050556
n_samples, n_features:  (2476, 595)
iterator started...
file processing: 2016-01-29.txt
final count = 2476

done in (sec):  21.2146019936
n_samples, n_features:  (2476, 2510)
iterator started...
file processing: 2016-01-29.txt
final count = 2476

done in (sec):  19.9292199612
n_samples, n_features:  (2476, 6203)
"""

# check context
tr -cd '\11\12\15\40-\176' # ascii printables
cut -f 13 news_tokenized_20160131/2016-01-*|tr '\n' ' '|sed 's/\.\s/\n/g' |sed 's/?\s/?\n/g' |grep -i "hillary_rodham_clinton_per"|sed 's/ \S\+_\(per\|loc\|org\|other\)_|//g' |less
cut -f 13 news_tokenized_20160131/2016-01-*|tr '\n' ' '|sed 's/\.\s/.\n/g' |sed 's/?\s/?\n/g' | sed 's/\."/\."\n/g' |python ~/projects/tweet/tweetNews/scripts/simpleGrep.py "powerball lottery jackpot million tickets ticket  winners winning drawing billion numbers prize   winner  odds    sales   robinson        tennessee       buy     sold"|sed 's/\S\+_\(per\|loc\|org\|other\)_|//g' |sort -t '     ' -k 1 -gr|uniq|less

cut -f1-10 ~/tmp/statsText_nyt_ent_100.txt|sed 's/\S\+_\(per\|loc\|org\|other\)_|//g'|sed 's/_|//g'|unidecode -e utf-8|less



Name conventions:

./data/
news_tokenized/
data pickle: ID.pickle
inits pickle: inits_ID_K.pickle


./out/
plsa_ID_K.pickle
out_ID_K.txt
out_ID_Kinits.txt
statsText_ID_K.txt
statsText_ID_Kinits.txt
"""
[jwang112@dmserv4 scripts]$ time python vectorize_news.py ../data/news_tokenized_20160307/ 2016-01-15 2016-01-22 ../data/20160115_0122.pickle
iterator started...
file processing: 2016-01-22.txt
final count = 848

done in (sec):  11.4629380703
n_samples, n_features:  (848, 14355)
iterator started...
file processing: 2016-01-22.txt
final count = 848

done in (sec):  4.69848108292
n_samples, n_features:  (848, 432)
iterator started...
file processing: 2016-01-22.txt
final count = 848

done in (sec):  5.41432404518
n_samples, n_features:  (848, 525)
iterator started...
file processing: 2016-01-22.txt
final count = 848

done in (sec):  4.80391407013
n_samples, n_features:  (848, 312)
iterator started...
file processing: 2016-01-22.txt
final count = 848

done in (sec):  9.07759308815
n_samples, n_features:  (848, 1269)
iterator started...
file processing: 2016-01-22.txt
final count = 848

done in (sec):  8.37045097351
n_samples, n_features:  (848, 4645)

real    0m49.846s
user    0m30.705s
sys     0m0.602s
"""
"""
[jwang112@dmserv4 scripts]$ time python vectorize_news.py ../data/news_tokenized_20160307/ 2016-01-23 2016-01-30 ../data/20160123_0130.pickle
iterator started...
file processing: 2016-01-30.txt
final count = 913

done in (sec):  14.2292890549
n_samples, n_features:  (913, 15071)
iterator started...
file processing: 2016-01-30.txt
final count = 913

done in (sec):  6.2032790184
n_samples, n_features:  (913, 455)
iterator started...
file processing: 2016-01-30.txt
final count = 913

done in (sec):  6.03657102585
n_samples, n_features:  (913, 573)
iterator started...
file processing: 2016-01-30.txt
final count = 913

done in (sec):  6.19543385506
n_samples, n_features:  (913, 312)
iterator started...
file processing: 2016-01-30.txt
final count = 913

done in (sec):  9.34933996201
n_samples, n_features:  (913, 1340)
iterator started...
file processing: 2016-01-30.txt
final count = 913

done in (sec):  8.27902388573
n_samples, n_features:  (913, 4995)

real    0m56.781s
user    0m34.631s
sys     0m0.552s

"""
