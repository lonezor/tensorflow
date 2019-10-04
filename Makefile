all:
	./generate_model.py
	./evaluate_model.py

clean:
	rm -f checkpoint*
