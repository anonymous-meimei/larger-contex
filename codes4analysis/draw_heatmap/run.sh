cat $1 | python diagLong-Short.py

find=".csv"
replace=""

cd heatmap
for i in `ls *.csv`
do
	echo $i
	result=${i//$find/$replace} 
	cat $i | python ../draw_heatmap.py --png $result.png
done
sz *.png
cp *.png /home/ubuntu/myLatex/TACL2020-NER/pic/longcontex/heatmap/
