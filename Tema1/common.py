def extract_constants(tokens):
    new_tokens = tokens[:-2]
    new_tokens = new_tokens[0::2]            

    letters = ['x', 'y', 'z']

    constants = []
    letters_dict = {}

    for i in range(len(new_tokens)):
        token = new_tokens[i]
        
        if tokens[2*i - 1] == '-':
            token = "-" + token

        letters_dict[token[-1:]] = token[:-1]

    for i in range(len(letters)):
        val = 0.0        
        letter_value = letters_dict.get(letters[i])
        if letter_value == "-":
            val = -1.0
        elif letter_value == "":
            val = 1.0
        elif letter_value:
            val = float(letter_value)
        constants.append(val)

    return constants


def parse_lines(lines):
    constants = []
    r = []

    for i in range(len(lines)):
        tokens = lines[i].split()
        r.append(float(tokens[-1]))

        constants.extend(extract_constants(tokens))


    return (constants, r)

def get_values_array():
    file = open("ecuatii.txt")
    lines = file.read().splitlines()

    return parse_lines(lines)

def print_values(mat):
    print("Solutiile sunt:")

    print("x = %f" % mat[0][0])
    print("y = %f" % mat[1][0])
    print("z = %f" % mat[2][0])
