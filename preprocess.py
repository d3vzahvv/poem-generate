def split(file="poetry_7.txt"):
    data = open(file, encoding="utf-8").read()
    dataSplit = " ".join(data)

    with open("split.txt", "w", encoding="utf-8") as f:
        f.write(dataSplit)