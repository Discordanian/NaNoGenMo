from textgenrnn import textgenrnn

textgen = textgenrnn()
# textgen.generate()
# textgen.train_from_file("./datasets/hacker-news-2000.txt", num_epochs=1)
textgen.train_from_file("datasets/RudyardKipling.txt", num_epochs=10)
textgen.generate()
textgen.generate(3, temperature=1.0)
# textgen_2 = textgenrnn('/weights/hacker_news.hdf5')
textgen_2.generate(3, temperature=1.0)
