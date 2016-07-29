eventID=$1
#time python eknot_stats_i.py ../data/20160107_0114.pickle ../output/plsa_20160107_0114_80.pickle /home/wtong8/NewsTwitter/tweets/ text $eventID ../output/summary_20160107_0114_80_$eventID.pickle > ../output/summary_20160107_0114_80_$eventID.txt
time python eknot_stats_i.py ../data/20160207_0213.pickle ../output/plsa_20160207_0213_70.pickle /home/wtong8/NewsTwitter/tweets/ text $eventID ../output/summary_20160207_0213_70_$eventID.pickle > ../output/summary_20160207_0213_70_$eventID.txt
