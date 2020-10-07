import re
from indexer.corpus_indexer import check_korean

class NullCleanser:
    def __init__(self):
        pass

    @staticmethod
    def cleanse(x):
        return x


class Cleanser:
    def __init__(self):
        delete_tokens = ['\t', '\r', '\x1e', '¤', '«', '»', '®', '°', '\xad', 'Ν', 'Φ', 'α', '‰', '‱', '','β', '\u2000',
                 '\u2003','\u200b','ủ','ừ','γ','μ','ν','π','Б','И', 'К', 'Р', 'Ф', 'ב', 'ו', 'ט', 'כ', 'ל', 'מ', 'ע',
                 'ᄎ', '⁰',]
        danwui = ['㎃', '㎈', '㎉', '㎍', '㎎', '㎏', '㎐', '㎑', '㎒', '㎓', '㎔', '㎕', '㎖', '㎗', '㎘', '㎚', '㎛',
          '㎜', '㎝', '㎞', '㎟', '㎠', '㎡', '㎢', '㎣', '㎤', '㎥', '㎦', '㎧', '㎨', '㎩', '㎫', '㎬', '㎰', '㎲', '㎳',
          '㎸', '㎹', '㎼', '㎽', '㎾', '㎿', '㏃', '㏄', '㏅', '㏈', '㏊', '㏓', '㏖', '㏜', ]
        todots = ['━', '│', '┃', '╗', '□', '▣', '▦', '▨', '▪', '△', '▴', '▷', '▸', '▹', '▼', '▽', '◇', '◈', '◉', '○',
          '◎', '●', '◦', '◯', '◾', '☁', '☂', '★', '☆', '☎', '☛', '☞', '☼', '♡', '♣', '♥', '♪', '♭', '✔', '✕', '✪',
          '❍', '❑', ]

        self.changes = {'·': '·',
           'à': 'a',
           'á': 'a',
           'ä': 'a',
           'é': 'e',
           'ê': 'e',
           'ì': 'i',
           'ö': 'o',
           '÷': '%',
           'ù': 'u',
           'ā': 'a',
           'ą': 'a',
           'ž': 'z',
           '˙': "'",
           '΄': "'",
           'Χ': 'X',
           "ᆞ": '·',
           "∙": '·',
           '‧': '·',
           "•": '·',
           "√": '·',
           "․": '.',
           "′": "'",
           "″": "'",
           '∼': '~',
           '∽': '~',
           '～': '~',
           '％': '%',
           '｝': '}',
           '‑': '-',
           '–': '-',
           '―': '-',
           '‘': "'",
           '’': "'",
           '‛': "'",
           '“': "'",
           '”': "'",
           '＇': "'",
           '！': '!',
           '＂': "'",
           '＃': '#',
           '‣': '·',
           '‥': '…',
           'ː': ':',
           '１': '1',
           '２': '2',
           '４': '4',
           '６': '6',
           'ｅ': 'e',
           'ｇ': 'g',
           'ｓ': 's',
           'ｍ': 'm',
           'ｔ': 't',
           'ｘ': 'x',
           'Ａ': 'A',
           'Ｋ': 'K',
           'Ｘ': 'X',
           '＋': '+',
           '，': ',',
           '－': '-',
           '．': '.',
           '／': '/',
           '（': '(',
           '）': ')',
           '｝': '}',
           '～': '~',
           '：': ':',
           '；': ';',
           '＜': '<',
           '＝': '=',
           '＞': '>',
           '？': '?',
           '＇': "'",
           '０': '0',
           '〃': "'",
           '〈': "<",
           '〉': ">",
           '《': '<',
           '》': '>',
           '「': '<',
           '」': '>',
           '『': '<',
           '』': '>',
           '〔': '<',
           '〕': '>',
           '≪': '<',
           '｢': '<',
           '｣': '>',
           '：': ':',
           '；': ';',
           '＜': '<',
           '＝': '=',
           '＞': '>',
           '［': '<',
           '］': '>',
           '＿': '_',
           '①': '1',
           '②': '2',
           '③': '3',
           '④': '4',
           '⑤': '5',
           '⑥': '6',
           '⑦': '7',
           '⑧': '8',
           '⑨': '9',
           '⑩': '10',
           '⑪': '11',
           '⑫': '12',
           '⑬': '13',
           '⓵': '1',
           '⓶': '2',
           '⓹': '5',
           '⓺': '6',
           '➀': '1',
           '➁': '2',
           '➂': '3',
           '➃': '4',
           '➄': '5',
           '➅': '6',
           '➈': '9',
           }

        self.delete_tokens = re.compile('['+''.join(delete_tokens)+']')
        self.measures = re.compile('['+''.join(danwui)+']')
        self.todots = re.compile(r'['+''.join(todots)+']')
        self.special_symbols = re.compile(r'([^ ?!.,a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ가-힣])')

    def cleanse_special_symbols(self,x):
        x = self.delete_tokens.sub('',x)
        x = self.measures.sub('㎃',x)
        x = self.todots.sub('·',x)
        x = self.special_symbols.sub(r' \1  ',x)
        for i in self.changes.keys():
            x = x.replace(i,self.changes[i])
        x = re.sub(' {2,}',' ',x)
        x = re.sub(r'\n{2,}',r'\n',x)
        return x


class GeniusClenser(Cleanser):
    def __init__(self):
        super(GeniusClenser, self).__init__()
        self.stlist = ['title','chorus', 'post-chorus', 'pre-chorus', 'bridge', 'intro', 'break', 'interlude', 'hook', 'rap',
                  'verse', 'outro']
        self.not_structure = re.compile('\[(?!{}).*\]'.format('|'.join(self.stlist)))
        self.struct_pattern = re.compile(r'\[[^\[]*\](\n|.)*?(?=\[)')

    @staticmethod
    def convert_title(title):
        if '(' in title:
            first, second = title.split('(', maxsplit=1)
            fm = re.match('[^ㄱ-ㅎㅏ-ㅣ가-힣]+', first)
            sm = re.match('[^a-zA-Z]+', second)
            if fm and sm and fm.span()[-1] == len(first) and sm.span()[-1] == len(second):
                target = second[:-1]
                return re.sub('\).+', '', target)
        return title

    def cleanse(self, lyrics, istitle):
        if istitle:
            x = self.convert_title(lyrics)
            x = re.sub('\(.+','',x).strip()
            return x.lower()
        if not check_korean(lyrics):
            return ''
        x = re.sub(r'\n\[everysing\] 고음의 끝이 안보이는(.|\n)*Stone Music Entertainment\n￼\n', '', lyrics)
        x = self.delete_tokens.sub('',x)
        x = re.sub(r'\]\n+',r'] ',x)
        x = re.sub('\.{2,}', '…', x)
        # 다하고
        x = re.sub(r' +\n', r'\n', x)
        x = re.sub(r'[*\-_.]', '_', x)
        x = re.sub(r'_{2,}', '__', x)
        x = re.sub(r'([*\-_.?!]){2,}', r'\1\1', x)
        x = re.sub(r'\n+', r'. ', x)
        return x.lower()

    def refine_struct(self, lyrics):
        # chorus
        lyrics = re.sub(r'\[chungha\]', '', lyrics)
        lyrics = re.sub(r'\[(extended )?cho?r?u[^\[]*?\]', '[chorus]', lyrics)
        lyrics = re.sub(r'\[(extended )?cho?r?u[^\[]*?\]', '[chorus]', lyrics)


        # post chorus
        lyrics = re.sub(r'\[post?[^\[]*chorus[^\[]*?\]', '[post-chorus]', lyrics)
        lyrics = re.sub(r'\[post?[^\[]*chrous[^\[]*?\]', '[post-chorus]', lyrics)

        # pre chorus
        lyrics = re.sub(r'\[pre[^\[]*?chor?us[^\[]*?\]', '[pre-chorus]', lyrics)
        lyrics = re.sub(r'\[pre[^\[]*?chrous[^\[]*?\]', '[pre-chorus]', lyrics)
        lyrics = re.sub(r'\[re[^\[]*?chorus[^\[]*?\]', '[pre-chorus]', lyrics)

        # bridge
        lyrics = re.sub(r'\[[^\[]*?bridge[^\[]*?\]', '[bridge]', lyrics)
        lyrics = re.sub(r'\[brigde\]', '[bridge]', lyrics)
        lyrics = re.sub(r'\[birdge\]', '[bridge]', lyrics)
        # break
        lyrics = re.sub(r'\[break[^\[]*?\]', '[break]', lyrics)

        # intro
        lyrics = re.sub(r'\[intro.*?\]', '[intro]', lyrics)
        # interlude
        lyrics = re.sub(r'\[interlude[^\[]*?\]', '[interlude]', lyrics)

        # hook
        lyrics = re.sub(r'\[(pre\\*\-|post\\\-)?hoo?k?[^\[]*\]', '[hook]', lyrics)
        lyrics = re.sub(r'\[pre-hoo\]', '[hook]', lyrics)

        # rap
        lyrics = re.sub(r'\[rap[^\[]*?\]', '[rap]', lyrics)

        # verse
        lyrics = re.sub(r'\[[^\[]*?verse.*?\]', '[verse]', lyrics)
        lyrics = re.sub(r'\[[^\[]*?(vefrse|veres|verrse|vers|vese|vesre)[^\[]*?\]', '[verse]', lyrics)

        # outro
        lyrics = re.sub(r'\[ending[^\[]*?\]', '[outro]', lyrics)
        lyrics = self.not_structure.sub('',lyrics)
        return lyrics + '[]'

    def struct(self, lyrics):
        def tag_lyrics(raw_phrase):
            temp = raw_phrase.split(']')
            return [temp[0][1:], temp[1]]
        lyrics = self.refine_struct(lyrics)
        res = self.struct_pattern.finditer(lyrics)
        structured = []
        try:
            for i in res:
                l, r = i.span()
                structured.append(tag_lyrics(lyrics[l:r]))
        except:
            print(lyrics)
        return structured


class Lyrics_Cleanser(Cleanser):
    def __init__(self):
        super(Lyrics_Cleanser, self).__init__()
        url = '(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.(com|org|net|co\.kr)'
        patterns = []
        patterns.append(re.compile(r'\r\n *\d{1,2}(절|\.|\)|:)?\)?\.? *(?=\r\n)'))
        patterns.append(re.compile(r'(?<=\r\n) *\d{1,2}(절|\.|\)|:) *'))
        patterns.append(re.compile(r'\(? *(피아노|코러스|노래|전주|간주|후주|유형|장르|발매|기획|작사자|작곡자|편곡자|가수명|작사|작곡|편곡|믹싱|마스터링|비트 메이킹|제목|앨범정보|연주정보|연주시간).*[\–=:\-].+(?=\r\n)'))
        patterns.append(re.compile(r'.+(작사|작곡|편곡)\r\n'))
        patterns.append(re.compile(r'\r\n *(assistant|mixed by|mixed at|guitars|recorded by|recorded at|작사|작곡|편곡|guitar|keyboard|background|strings|piano|drum).+(?=\r\n)'))
        patterns.append(re.compile(r'\r\n가사 +'))
        patterns.append(re.compile(r'\(.+:.+\)'))
        patterns.append(re.compile(
            r'.*(song|mixed|written|arrangged|produced|arranged|compose|guitars|composed|rap|lyrics|chorus|vocal|arrange|lyric|keyboard|piano|guitar|programming|bass|drum|rep)[^가-힣]+by.+'))
        patterns.append(re.compile(r'가사 ?:'))
        patterns.append(re.compile(r'(팀명|곡 제목) :.+'))
        patterns.append(re.compile(r'\+\+\r\n작사(.|\n)+'))
        patterns.append(re.compile(r'(원곡명|원곡출판자|국내출판대행사):'))
        patterns.append(re.compile(r'\r\n ?feat\..+'))
        patterns.append(
            re.compile(r'.*(vocal|verse|hook|sabi|narration|intro|outro|rap|bridge|chorus|song)[_ ]*\d{0,2}[\)\/:ㅡ]+'))
        patterns.append(re.compile(r'\(.+(\d절|간주|작사|작곡|편곡|반복|전주|후렴|한번더).+\)'))
        patterns.append(re.compile(r'다양한 해외힙합곡을 감상하고 싶다면(.|\n)+'))
        patterns.append(re.compile(r'\(? *(\*|x|X)\d *\)?'))
        patterns.append(re.compile(r'nit music maniaclub...angel(.|\n)+'))
        patterns.append(re.compile(r'가사 오류 제보(.|\n)+'))
        patterns.append(re.compile(r'\*+ ?(반복|repeat)'))
        patterns.append(re.compile(r'출처 ?[:-].+'))
        patterns.append(re.compile(r'.+가사출처(.|\n)+'))
        patterns.append(re.compile(r'\[.+\]'))
        patterns.append(re.compile(r'.+> *'))
        patterns.append(re.compile(r'[-=*]{15,}(.|\n)+'))
        patterns.append(re.compile(r'마음의 양식을 주는 명언:(.|\n)+'))
        patterns.append(re.compile(r'.+음반구입합시다.+'))
        patterns.append(re.compile(r'제가 올린것만 수정이 됩니당(.|\n)+'))
        patterns.append(re.compile(r'가사 틀린 부분이 있으면(.|\n)+'))
        patterns.append(re.compile(r'오직 제가 올린 것만 수정이 됩니다. ^^;'))
        patterns.append(re.compile(r'사랑에 대한 짧은 명언:(.|\n)+'))
        patterns.append(re.compile(r'.+틀린 ?가사(.|\n)+'))
        patterns.append(re.compile(r'.+{}.+'.format(url)))
        patterns.append(re.compile(r'B Y . 샹 ㄴㅖ 지 훈 수 현(.|\n)+'))
        patterns.append(re.compile(r'by.야마(.|\n)+'))
        patterns.append(re.compile(r'\[hani\]'))
        patterns.append(re.compile(r'※'))
        patterns.append(re.compile(r'\[일랜시아 엘서버 아이디\]\r\n변미선사랑해\r\n'))
        patterns.append(re.compile(r'\*(?!\*)'))
        self.patterns = patterns
        translated = []
        translated.append(re.compile(r'[┌│└]'))
        self.translated = translated
        notuse = []
        notuse.append(re.compile(r'안녕하세요 사연 읽어주는 남자, 사연남입니다.'))
        self.notuse = notuse

    def convert_title(self, title):
        if '(' in title:
            first, second = title.split('(', maxsplit=1)
            fm = re.match('[^ㄱ-ㅎㅏ-ㅣ가-힣]+', first)
            sm = re.match('[^a-zA-Z]+', second)
            if fm and sm and fm.span()[-1] == len(first) and sm.span()[-1] == len(second):
                target = second[:-1]
                return re.sub('\).+', '', target)
        return title

    def check_garbage(self, lyrics):
        for i in self.translated:
            if len(i.findall(lyrics)) >10:
                return True
        for i in self.notuse:
            if i.findall(lyrics):
                return True
        return False

    def cleanse(self, x, istitle):
        if istitle:
            x = self.convert_title(x)
            x = re.sub('\(.+','',x).strip()
            return x
        else:
            if self.check_garbage(x):
                return ''
            for i in self.patterns:
                x = i.sub('', x)
            x = re.sub(r'\r\n +', r'\r\n', x)
            x = re.sub(r'(\r\n)+', r'\r\n', x)
            x = re.sub(r'\n{2,}',r'\n', x)
            x = re.sub(r'\n +', r'\n', x)
            x = re.sub(r'\.\r',r'\r',x)
            x = re.sub('\.{2,}','…', x)
            # 다하고
            x = re.sub(r' +\r\n',r'\r\n',x)
            x = re.sub(r'[*\-_.]','_',x)
            x = re.sub(r'_{2,}','__',x)
            x = re.sub(r'([*\-_.?!]){2,}',r'\1\1',x)
            x = re.sub(r'^(\r\n)','',x)
            x = re.sub(r'(\r\n)+',r'. ',x)
            return x


class News_Cleanser(Cleanser):
    def __init__(self):
        super(News_Cleanser, self).__init__()
        patterns = []
        patterns.append(re.compile(r'\d{,4} ?\. ?\d{,2} ?\. ?\d{,2} ?.+확 달라진 연합뉴스 웹을.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,4}[0-9a-zA-Z\-\_\.]* ?네이버에서 뉴시스 채널 구독하기.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,4} ?(기자)? ?[0-9a-zA-Z\-\_\.]* ?ⓒ 세상을 보는 눈 글로벌 미디어 '))
        patterns.append(re.compile(r'\d+\-\d+네이버에서 헤럴드경제.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,6} ?(기자 )?[0-9a-zA-Z\-\_\.]*저작권자 ⓒ 성공을 꿈꾸는 사람들의 경제.+'))
        patterns.append(re.compile(r'[0-9a-zA-Z\-\_\.]* 기사입니다.정기구독한국경제매거진한경비즈니스.+'))
        patterns.append(re.compile(r'네이버 홈에서 채널 구독하기뭐 하고 놀까\? 흥 쇼미더뉴스! 오늘 많이 본 뉴스영상.*'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,3} 기자 [0-9a-zA-Z\-\_\.]+.+한국경제신문·연합뉴스 기사입니다.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{2,4} 넥스트데일리 기자 ?[0-9a-zA-Z\-\_\.]+'))
        patterns.append(re.compile(r'디지털뉴스부기자.+디지털타임스를 구독해주세요'))
        patterns.append(re.compile(r'\.[^.]+[0-9a-zA-Z\-\_\.]+[^.]+위 기사는 한국언론진흥재단의 지원.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,3} ?(한경닷컴 )?기자 [0-9a-zA-Z\-\_\.]+한경닷컴 바로가기모바일한경 구독신청자세히.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,3} ?기자 [0-9a-zA-Z\-\_\.]+한경닷컴 바로가기모바일한경 구독신청자세히.+'))
        patterns.append(re.compile(r'(헤럴드경제가 )?(freiheitheraldcorp.com )?연재를 새롭게 시작합니다.+많은 제보 기다립니다\.'))
        patterns.append(re.compile(r'ⓒ매일신문 - www.imaeil.com'))
        patterns.append(re.compile(r'[0-9a-zA-Z\-\_\.]+모바일한경은 프리미엄 채널입니다.'))
        patterns.append(re.compile(r'[0-9a-zA-Z\-\_\.]+ [ㄱ-ㅎㅏ-ㅣ가-힣]{0,3} 기자.+저작권자 ⓒ 파이낸셜뉴스.+'))
        patterns.append(re.compile(r'startblock━━endblock'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,3} ?(기자)?저작권ⓒ 건강을 위한 정직한 지식.+'))
        patterns.append(re.compile(r'\d+-\d+.+Copyright ⓒ 스포츠서울.+'))
        patterns.append(re.compile(r'articlesplit.+돈이 보이는 리얼타임.+'))
        patterns.append(re.compile(r'데일리안.+ⓒ 데일리안.+'))
        patterns.append(re.compile(r'뉴스는 머니투데이.+'))
        patterns.append(re.compile(r'ⓒ 세상을 보는 눈.+'))
        patterns.append(re.compile(r'독자 퍼스트 언론 한겨레21 정기구독으로 응원하기!.+'))
        patterns.append(re.compile(r'이데일리 채널 구독하면 꿀잼가득 빡침해소!청춘뉘우스 ＜ⓒ종합 경제정보 미디어 이데일리.+'))
        patterns.append(re.compile(r'인스타 친추하면 연극표 쏩니다! 꿀잼가득 청춘뉘우스 ＜ⓒ종합 경제정보.+'))
        patterns.append(re.compile(r'＜ⓒ종합 경제정보 미디어 이데일리 - 무단전재 & 재배포 금지＞.+'))
        patterns.append(re.compile(r'네이버 홈에서 ‘이데일리’ 기사 보려면.+'))
        patterns.append(re.compile(r'텔레그램으로 서울경제 구독하기.+'))
        patterns.append(re.compile(r'Telegram으로 서울경제 뉴스를 실시간으로 받아보세요.+'))
        patterns.append(re.compile(r'[0-9a-zA-Z\-\_\.]+\(본 기사는 헤럴드경제로부터 제공받은 기사입니다.+'))
        patterns.append(re.compile(r'MSI는 총 7단계로 나뉜다. 1단계 매우 나쁨 2단계 나.+$'))
        patterns.append(
            re.compile(r'(온라인뉴스팀 sportskyunghyangkyunghyang.com)?©스포츠경향\(sports.khan.co.kr\) 무단전재 및 재배포 금지'))
        patterns.append(re.compile(r'모니터 기간과 대상 \d+년 \d월.+'))
        patterns.append(re.compile(r'저작권자? ?ⓒ.+'))
        patterns.append(re.compile(r'ⓒ 동아일보.+'))
        patterns.append(re.compile(r'[0-9a-zA-Z\-\_\.]* 연합뉴스 채널 구독하고.+'))
        patterns.append(re.compile(r'\.[^.]+[0-9a-zA-Z\-\_\.]+ © 주간경향.+'))
        patterns.append(re.compile(r'이다아경봇 기자.+'))
        patterns.append(re.compile(r'저작권자 오마이뉴스 무단 전재.+'))
        patterns.append(re.compile(r'[0-9a-zA-Z\-\_\.]* ?여러분의 제보를 기다립니다.'))
        patterns.append(re.compile(r'공감언론 뉴시스가 독자 여러분의 소중한 제보를 기다립니다.+'))
        patterns.append(re.compile(r'네이버 홈에서.+종합 경제정보 미디어 이데일리 - 무단전재.+'))
        patterns.append(re.compile(r'서울Biz 바로가기인기 무료만화네이버에서.+'))
        patterns.append(re.compile(r'당첨자 2222명!.+＜ⓒ종합 경제정보 미디어 이데일리.+'))
        patterns.append(re.compile(r'(\d+-\d+-\d+)? ?MobileAdNew center.+'))
        patterns.append(re.compile(r'(\d+-\d+-\d+)?서울Biz 바로가기인기.+'))
        patterns.append(re.compile(r'뉴스제보 이메일 카카오톡 전화- Copyrights ⓒ TV조선. 무단전재 및 재배포 금지 -.+'))
        patterns.append(re.compile(r'(인터뷰 전문 )?YTN \& YTN PLUS 무단전재 및 재배포 금지\n\t'))
        patterns.append(re.compile(r'이데일리.+ ＜ⓒ종합 경제정보 미디어 이데일리 - 무단전재 & 재배포 금지＞.*'))
        patterns.append(re.compile(r'ⓒ경제를 보는 눈 세계를 보는 창 아시아경제 무단전재 배포금지.*'))
        patterns.append(re.compile(r'뉴스제보 이메일 카카오톡 전화- Copyrights ⓒ TV조선.+'))
        patterns.append(re.compile(r'저작권자 © CBS 노컷뉴스.+'))
        patterns.append(re.compile(r'서울Biz 바로가기네이버에서 서울신문 구독하기ⓒ .+'))
        patterns.append(re.compile(r'네이버 메인에서 경향신문 받아보기두고 두고 읽는 뉴스인기 무료만화©경향신문.+'))
        patterns.append(re.compile(r'(연합뉴스 유튜브 )?네이버 홈에서 채널 구독하기'))
        patterns.append(re.compile(r'＜ⓒ종합 경제정보 미디어 이데일리 - 무단전재 & 재배포 금지＞'))
        patterns.append(re.compile(r'(뉴시스)?GoodNews paper ⓒ 국민일보 무단전재.+'))
        patterns.append(re.compile(r'Copyright ⓒ MBN 무단전재 및 재배포 금지.*'))
        patterns.append(re.compile(r' 뉴스제보 이메일 카카오톡.+ⓒ TV조선. 무단전재 및 재배포 금지.*'))
        patterns.append(re.compile(r'뉴스두고 두고 읽는 뉴스인기 무료만화©경향신문.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]{0,3} 기자 MBC 무단복제-재배포 금지.+'))
        patterns.append(re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]* ?[ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]* ?관련 뉴스·포토 보기 네이버메인에 ‘뉴스1채널’ 설정하기.+'))
        self.patterns = patterns
        delete_patterns = []
        delete_patterns.append(re.compile(r'SBS I&M 무단전재'))
        self.delete_patterns = delete_patterns

    def cleanse(self,x, istitle=False):
        for i in self.delete_patterns:
            if i.search(x):
                return ''
        for i in self.patterns:
            x = i.sub('',x)
        x = re.sub(r'(\.)([^ ])', r'\1 \2', x)
        x = re.sub(r'\(\)', '', x)
        x = re.sub(r'\. [ㄱ-ㅎㅏ-ㅣ가-힣]{2,4} ?기자 ?[0-9A-Za-z\-\_]+\.(com|net|org|co\.kr)', '.', x)
        x = self.cleanse_special_symbols(x)
        if istitle: return x
        elif len(x)<100 and '별세' in x:
            return ''
        elif len(x) < 50:
            return ''
        else:
            return x.lower()


class Poet_Cleanser(Cleanser):
    def __init__(self):
        super(Poet_Cleanser, self).__init__()

    def delete_unk(self,text):
        text = re.sub('(UNK)+ *\(([a-zA-Z,ㄱ-ㅎ가-힣]+)\)', r'\2', text)  # UNK 괄호 안에 한글 한글로 바꿈
        text = re.sub(r'\([^)]*?UNK[^)]*?\)', r'', text)  # 괄호 안에 UNK 있는 것들 삭제
        text = re.sub(r'(?<=( |\n))(UNK)+[^(UNK)](?=( |\n))', '', text)
        text = re.sub(r'(?<=( |\n))[^(UNK)](UNK)+(?=( |\n))', '', text)
        text = re.sub(r'(?<=( |\n))(UNK)+[^(UNK)](UNK)+(?=( |\n))', '', text)
        text = re.sub(r'UNK', '', text)
        return text

    def cleanse(self,x):
        x = self.cleanse_special_symbols(x)
        x = re.sub(r'\n','. ',x)
        return x.lower()