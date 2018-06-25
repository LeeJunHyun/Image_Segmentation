
for ((i=0;i<100;i++));do
	a='U_Net'
	python3 main.py --model_type=$a
	a='R2U_Net'
	python3 main.py --model_type=$a
	a='AttU_Net'
	python3 main.py --model_type=$a
	a='R2AttU_Net'
	python3 main.py --model_type=$a
	
done
