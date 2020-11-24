

def evaluate_word_PRF(y,y_pred,  item2idx_dic,test = False):
	y = sum(y,[])
	y_pred = sum(y_pred, [])
	# print('y_pred',y_pred)
	print()
	e_idx = 'E'
	s_idx = 'S'
	cor_num = 0
	yp_wordnum = y_pred.count(e_idx)+y_pred.count(s_idx)
	yt_wordnum = y.count(e_idx)+y.count(s_idx)
	start = 0
	for i in range(len(y)):
		if y[i] == e_idx or y[i] == s_idx:
			flag = True
			for j in range(start, i+1):
				if y[j] != y_pred[j]:
					flag = False
			if flag == True:
				cor_num += 1
			start = i+1

	P = 100*cor_num / float(yp_wordnum) if yp_wordnum > 0 else 0.0
	R = 100*cor_num / float(yt_wordnum) if yt_wordnum > 0 else 0.0
	F = 2 * P * R / (P + R) if yp_wordnum > 0 else 0.0
	# P = '%.2f'% P
	# R = '%.2f'% R
	# F = '%.2f'% F

	print('P: ', P)
	print('R: ', R)
	print('F: ', F)

	if test:
		return P,R,F
	else:
		return F