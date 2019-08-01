punctuation = [',', '\"', '(', ')', ';', '-', ':']
with open('uid2text.txt', 'w') as out:
    with open('metadata.txt', 'r') as f:
        for line in f:
            l1 = line.split('|')
            l2 = l1[0].split('/')
            string = ''.join(c for c in l1[1] if c not in punctuation).upper()[:-1]
            if string[-1] == '.':
                string = string[:-1]
            
            line =  l2[2].replace('.wav', '') + ' ' + string
            print (line)
            out.write(line + '\n')
