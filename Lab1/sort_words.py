def sort_words():
    file = open("Latin-Lipsum.txt", 'r')
    text = file.read().split()
    text = [word.strip('.').split(',') for word in text]

    text = sorted(text);
    print(text)

sort_words();