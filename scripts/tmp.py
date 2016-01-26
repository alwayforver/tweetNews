import re
tw_text = 'Halliburton acquires Baker Hughes https://t.co/b0lzLaVYXU | https://t.co/DQ4i0wlhe8'
tw_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tw_text)
print tw_text
s = tw_text.find("http://")
if s == -1:
    s = tw_text.find("https://")
if s != -1:
    tmp = tw_text[s:] 
    e = tmp.find(" ")
    if e == -1:
        e = len(tmp)
    tw_text = (tw_text[:s].strip()+ " " + tmp[e:].strip()).strip()
print tw_text
