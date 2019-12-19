

def break_lines(string, every=48):
    return '\n'.join(string[i:i+every] for i in range(0, len(string), every))