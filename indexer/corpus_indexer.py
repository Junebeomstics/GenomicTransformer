import argparse
from contextlib import contextmanager
from multiprocessing import Process, Pool
from indexer.encoder import *
from indexer.morph_tagger import *
from indexer.preprocess import *
from util.files import *
from indexer.encoder import SPEncoder

@contextmanager
def poolcontext(*args,**kwargs):
    pool=Pool(*args,**kwargs)
    yield pool
    pool.terminate()


def check_korean(sample):
    japanese = re.compile('/[一-龠]+|[ぁ-ゔ]+|[ァ-ヴー]+|[々〆〤]+/u')
    chinese = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    searched_japanese = japanese.findall(sample)
    searched_chinese = chinese.findall(sample)
    if len(searched_japanese) > 10 or len(searched_chinese) > 10:
        return False
    return bool(re.search(pattern=r".*[가-힣].*",string=sample))


class Indexer:
    def __init__(self, cleanser, encoder_type='SP', **kwargs):
        if encoder_type =='SP':
            self.enc = SPEncoder(**kwargs)
        elif encoder_type =='HF':
            self.enc = HFEncoder(**kwargs)
        self.cleanser = cleanser
        self.encoder_type=encoder_type

    def cleanse_encode(self, inp):
        return self.enc.encode(self.cleanser.cleanse(inp))

    def index_one(self,inp, out):
        raise NotImplementedError

    def corpus_encode(self, directory_path, inp_path, out_path):
        inp_path = os.path.join(directory_path, inp_path)
        procs = []
        fl = get_files(inp_path)
        out_dir = os.path.join(directory_path, out_path)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        for index, inp in enumerate(fl):
            basename, _ = os.path.splitext(inp)
            basename = os.path.basename(basename)
            out = os.path.join(out_dir, basename + '.pkl')
            proc = Process(target=self.index_one, args=(inp, out))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()


class LyricsIndexer(Indexer):
    def __init__(self, cleanser, encoder_type, use_morphs, **kwargs):
        super(LyricsIndexer, self).__init__(cleanser, encoder_type, **kwargs)
        if encoder_type =='SP':
            assert use_morphs == False
        self.use_morphs = use_morphs
        self.eot = self.enc.sep

    def lyrics_tostring(self, inp):
        title, lyrics = inp
        cleanser = self.cleanser
        tokenizer = self.enc.tokenizer
        title = str(title).lower()
        lyrics = str(lyrics).lower()
        if not check_korean(lyrics):
            return None
        t, l = cleanser.cleanse(title, istitle=True), cleanser.cleanse(lyrics, istitle=False)
        if self.use_morphs:
            t = tokenizer.text_to_morphs(str(t), True)
            l = tokenizer.text_to_morphs(str(l), True)
        if l: return t + '{} '.format(self.eot) + l + '\n'
        else: return ''


class PlainLyricsIndexer(LyricsIndexer):
    def __init__(self, encoder_type, use_morphs, **kwargs):
        super(PlainLyricsIndexer, self).__init__(Lyrics_Cleanser(), encoder_type, use_morphs, **kwargs)

    def index_one(self,inp, out):
        df = pd.read_csv(inp)
        tokenized_lyrics = list(map(self.lyrics_tostring, zip(df['title'].tolist(), df['lyrics'].tolist())))
        tokenized_lyrics = list(set([i for i in tokenized_lyrics if i]))
        indexed = [self.enc.encode(i.rstrip()) for i in tokenized_lyrics]
        lengths = [len(i) for i in indexed]
        df = pd.DataFrame({'texts': indexed, 'lens': lengths})
        df.to_pickle(out)


class StructuredLyricsIndexer(LyricsIndexer):
    def __init__(self, encoder_type, use_morphs, **kwargs):
        super(StructuredLyricsIndexer, self).__init__(GeniusClenser(), encoder_type, use_morphs, **kwargs)
        self.struct_dic = get_structure_dict()

    def index_one(self, inp, out):
        def index_single_song(song):
            struct_indexed = []
            lyrics_indexed = []
            for i, struct in enumerate(song):
                str_indexed = self.struct_dic[struct[0]]
                is_bos = False if i else True
                is_eos = True if i == len(song) - 1 else False
                lindexed = self.enc.encode(struct[1].strip(), is_bos, is_eos)
                struct_indexed.extend([str_indexed]*len(lindexed))
                lyrics_indexed.extend(lindexed)
            return struct_indexed, lyrics_indexed

        df = pd.read_pickle(inp)
        df = df[df.st_unst == 'structured']
        tokenized_lyrics = list(map(self.lyrics_tostring, zip(df['song_title'].tolist(), df['song_lyrics'].tolist())))
        tokenized_lyrics = list(set(['[title] '+i for i in tokenized_lyrics if i]))
        structured_lyrics = [self.cleanser.struct(i) for i in tokenized_lyrics]
        temp = [index_single_song(i) for i in structured_lyrics]
        struct = list(map(lambda x: x[0], temp))
        indexed = list(map(lambda x: x[1], temp))
        lengths = [len(i) for i in indexed]
        df = pd.DataFrame({'texts': indexed, 'structure':struct, 'lens': lengths})
        df.to_pickle(out)



class HFIndexer(Indexer):
    def __init__(self, directory_path, prefix, vocab_size = 10000,
                 tokenizer_class=Mecab_Tokenizer, encoder_class=tokenizers.SentencePieceBPETokenizer,
                 jamo=False,
                 ):
        super(HFIndexer, self).__init__()
        self.directory_path = directory_path
        self.prefix = prefix
        self.encoder_class = encoder_class
        self.vocab_size = vocab_size
        self.jamo = jamo
        self.tokenizer = tokenizer_class(space_symbol='쀍', jamo=jamo)
        self.encoder = self.load_encoder(encoder_class, directory_path, prefix)

    def decode(self, indexed):
        return self.tokenizer.morphs_to_text(self.encoder.decode(indexed))

    def encode(self,text):
        morphed = self.tokenizer.text_to_morphs(str(text), True)
        return self.encoder.encode(morphed).ids



    def index_one(self, inp, out):
        tokenized_lyrics = self.tokenized_lists(inp)
        indexed = [self.encoder.encode(i.rstrip()).ids for i in tokenized_lyrics]
        lengths = [len(i) for i in indexed]
        df = pd.DataFrame({'texts': indexed, 'lens': lengths})
        df.to_pickle(out)

    def corpus_encode(self, inp_path, out_path):
        if not self.istrained:
            enc = self.learn_encoder(inp_path)
            enc.save(self.directory_path, self.encoder_filename)
            self.encoder = self.load_encoder(self.encoder_class,self.directory_path,self.encoder_filename)
        inp_path = os.path.join(self.directory_path, inp_path)
        procs = []
        fl = get_files(inp_path)
        out_dir = os.path.join(self.directory_path, out_path)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        for index, inp in enumerate(fl):
            basename, _ = os.path.splitext(inp)
            basename = os.path.basename(basename)
            out = os.path.join(out_dir, basename+'.pkl')
            proc = Process(target=self.index_one, args=(inp, out))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

    def tokenized_lists(self, file_path):
        """
        should be implemented in child class
        :param file_path:
        :return:
        """
        raise NotImplementedError


class HFLyricsIndexer(HFIndexer):
    def __init__(self, directory_path, encoder_filename, **kwargs
                 ):
        super(HFLyricsIndexer, self).__init__(directory_path, encoder_filename, **kwargs)
        self.cleanser = Lyrics_Cleanser()

    def lyrics_tostring(self, inp):
        title, lyrics = inp
        cleanser = self.cleanser
        tokenizer = self.tokenizer
        title = str(title).lower()
        lyrics = str(lyrics).lower()
        if not check_korean(lyrics):
            return None
        t, l = cleanser.cleanse(title, istitle=True), cleanser.cleanse(lyrics, istitle=False)
        t = tokenizer.text_to_morphs(str(t), True)
        l = tokenizer.text_to_morphs(str(l), True)
        if self.encoder_class == tokenizers.BertWordPieceTokenizer:
            if l: return t + ' １ ' + l + '\n'
            else: return ''
        else:
            if l: return ' SOT ' + t + ' SOL ' + l + ' EOL \n'
            else: return ''

    def tokenized_lists(self, filepath):
        df = pd.read_csv(filepath)
        tokenized_lyrics = list(map(self.lyrics_tostring, zip(df['title'].tolist(), df['lyrics'].tolist())))
        tokenized_lyrics = list(set([i for i in tokenized_lyrics if i]))
        return tokenized_lyrics


class HFNewsIndexer(HFIndexer):
    def __init__(self, directory_path, encoder_filename, **kwargs
                 ):
        super(HFNewsIndexer, self).__init__(directory_path, encoder_filename, **kwargs)
        self.cleanser = News_Cleanser()

    def news_tostring(self, inp):
        title, contents = inp
        cleanser = self.cleanser
        tokenizer = self.tokenizer
        title = str(title)
        contents = str(contents)
        t, l = cleanser.cleanse(title,True), cleanser.cleanse(contents,False)
        t = tokenizer.text_to_morphs(str(t), True)
        l = tokenizer.text_to_morphs(str(l), True)

        if self.encoder_class == tokenizers.BertWordPieceTokenizer:
            if l: return t + ' １ ' + l + '\n'
            else: return ''
        else:
            if l: return ' SOT ' + t + ' SOL ' + l + ' EOL \n'
            else: return ''

    def tokenized_lists(self, filepath):
        df = pd.read_pickle(filepath)
        tokenized_lyrics = list(map(self.news_tostring, zip(df['title'].tolist(), df['contents'].tolist())))
        tokenized_lyrics = list(set([i for i in tokenized_lyrics if i]))
        return tokenized_lyrics


class HFPoetIndexer(HFIndexer):
    def __init__(self, directory_path, encoder_filename, **kwargs):
        super(HFPoetIndexer, self).__init__(directory_path, encoder_filename, **kwargs)
        self.cleanser = Poet_Cleanser()

    def poet_tostring(self, inp):
        title, contents = inp
        cleanser = self.cleanser
        tokenizer = self.tokenizer
        title = str(title)
        contents = str(contents)
        t, l = cleanser.cleanse(title), cleanser.cleanse(contents)
        t = tokenizer.text_to_morphs(str(t), True)
        l = tokenizer.text_to_morphs(str(l), True)
        if self.encoder_class == tokenizers.BertWordPieceTokenizer:
            if l: return t + ' １ ' + l + '\n'
            else: return ''
        else:
            if l: return ' SOT ' + t + ' SOL ' + l + ' EOL \n'
            else: return ''

    def tokenized_lists(self, filepath):
        df = pd.read_csv(filepath,encoding = 'cp949')
        tokenized_lyrics = list(map(self.poet_tostring, zip(df['title'].tolist(), df['main'].tolist())))
        tokenized_lyrics = list(set([i for i in tokenized_lyrics if i]))

        return tokenized_lyrics


class HFTERIndexer(HFIndexer):
    def __init__(self, directory_path, encoder_filename, **kwargs
                 ):
        super(HFTERIndexer, self).__init__(directory_path, encoder_filename, **kwargs)

    def index_one(self, inp, out):
        tokenized_lyrics, labels = self.tokenized_lists(inp)
        indexed = [self.encoder.encode(i.rstrip()).ids for i in tokenized_lyrics]
        lengths = [len(i) for i in indexed]
        df = pd.DataFrame({'texts': indexed, 'lens': lengths, 'labels': labels})
        df.to_pickle(out)

    def tostring(self, inp):
        tokenizer = self.tokenizer
        inp = str(inp).lower()
        t = tokenizer.text_to_morphs(str(inp), True)
        if self.encoder_class == tokenizers.BertWordPieceTokenizer:
            if t: return t + ' \n'
            else: return ''
        else:
            if t: return ' SOT ' + t + ' EOL \n'
            else: return ''

    def tokenized_lists(self, filepath):
        df = pd.read_excel(filepath)
        tokenized_lyrics = list(map(self.tostring, df.Sentence.tolist()))
        tokenized_lyrics = list([i for i in tokenized_lyrics if i])
        labels = df.Emotion.tolist()
        return tokenized_lyrics, labels


class SPTERIndexer(Indexer):
    def __init__(self, directory_path, *args, **kwargs):
        super(SPTERIndexer, self).__init__(directory_path, NullCleanser())

    def index_one(self, inp, out):
        df = pd.read_excel(inp)
        indexed = [self.cleanse_encode(j) for j in df['Sentence'].tolist()]
        lengths = [len(i) for i in indexed]
        df = pd.DataFrame({'texts': indexed, 'lens': lengths, 'labels': df['Emotion'].tolist()})
        df.to_pickle(out)


class LMIndexer:
    def __init__(self, root, prefix, encoder_class=tokenizers.SentencePieceBPETokenizer, vocab_size=30000):
        self.encoder = self.load_encoder(encoder_class, root, prefix)
        self.map_dic, self.inv_dic = self.load_mapper(root, prefix)
        self.vocab_size = vocab_size
        self.root = root
        self.prefix = prefix

    def decode(self, indexed, map=True):
        if map and self.map_dic is not None:
            indexed = self.reverse_map(indexed)
        return self.encoder.decode(indexed)

    def encode(self, text, map=True):
        encoded = self.encoder.encode(text).ids
        if map and self.map_dic is not None:
            encoded = self.convert_map(encoded)
        return encoded

    def convert_map(self, encoded):
        def convert_token(token):
            if key_type(token) in dic:
                return dic[key_type(token)]
            else:
                return len(dic) - 1
        assert self.map_dic is not None
        dic = self.map_dic
        key_type = type(list(self.map_dic)[0])
        converted = [convert_token(i) for i in encoded]
        return converted

    def reverse_map(self, encoded):
        return [int(self.inv_dic[i]) for i in encoded]

    def learn_encoder(self, path):
        self.encoder.train(path, vocab_size=self.vocab_size)
        self.encoder.save(self.root,self.prefix)

    def learn_mapper(self,encoded):
        prob, dic = self.count(encoded,self.vocab_size)
        inv_dict = dict(zip(dic.values(), dic.keys()))
        base_name = os.path.join(self.root, self.prefix)
        dic_name = base_name + '-dic.pkl'
        prob_name = base_name + '-probs.pkl'
        json.dump(prob, open(prob_name, 'w'))
        json.dump(dic, open(dic_name, 'w'))
        self.map_dic, self.inv_dic = dic, inv_dict

    def load_mapper(self, root, prefix):
        base_name = os.path.join(root, prefix)
        dic_name = base_name + '-dic.pkl'
        if os.path.exists(dic_name):
            dic = load_json(dic_name)
            inv_dic = dict(zip(dic.values(), dic.keys()))
        else:
            dic=None
            inv_dic=None
        return dic, inv_dic

    def load_encoder(self, encoder_class, directory_path, encoder_filename):
        base_name = os.path.join(directory_path, encoder_filename)
        if encoder_class == tokenizers.BertWordPieceTokenizer:
            vocab_name = base_name + '-vocab.txt'
            if os.path.exists(vocab_name):
                print('trained encoder loaded')
                self.istrained = True
                return encoder_class(vocab_name)
            else:
                self.istrained = False
                print('encoder needs to be trained')
                return encoder_class()
        else:
            vocab_name = base_name + '-vocab.json'
            merge_name = base_name + '-merges.txt'
            if os.path.exists(vocab_name) and os.path.exists(merge_name):
                print('trained encoder loaded')
                self.istrained = True
                if encoder_class == tokenizers.SentencePieceBPETokenizer:
                    return encoder_class(vocab_name, merge_name)
                else:
                    return encoder_class(vocab_name, merge_name, lowercase=True)
            else:
                self.istrained = False
                print('encoder needs to be trained')
                if encoder_class == tokenizers.SentencePieceBPETokenizer:
                    return encoder_class()
                else:
                    return encoder_class(lowercase=True)

    def count(self, tl, vocab_size=30000):
        import collections
        cnter = collections.Counter()
        cnter.update(tl)
        for i in range(vocab_size):
            if i not in cnter:
                cnter[i] = 1

        tot = 0
        cum_prob = [0]
        for i in cnter.most_common():
            tot += i[1]
        for i in cnter.most_common():
            cum_prob.append(cum_prob[-1] + i[1] / tot)
        cum_prob.pop(0)
        new_dict = dict([(int(old[0]), int(new)) for (new, old) in enumerate(cnter.most_common())])
        return cum_prob, new_dict


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory_path", type=str, default=r"../../data/bugs",
                        help='parent directory path')
    parser.add_argument("--inp_path", type=str,
                        help='directory where input data is stored')
    parser.add_argument("--encoder_filename", type=str, default=r"lyrics",
                        help='encoder will be stored with this name')
    parser.add_argument("--out_path", type=str,
                        help='directory path where encoded data is stored')
    parser.add_argument("--encoder_class", type=str, default="SP")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--tokenizer_class", type=str, default="mecab")
    parser.add_argument("--use-morphs", action='store_true')
    parser.add_argument("--subencoder_class", type=str, default="sentencepiecebpe")
    parser.add_argument("--indexer_type", type=str, default="lyrics")
    parser.add_argument("--split_jamo", action='store_true')
    return parser


def main():
    parser = get_parser()
    tokenizers_map = {'mecab': Mecab_Tokenizer}
    subencoder_map = {'sentencepiecebpe': tokenizers.SentencePieceBPETokenizer,
                   'wordpiece':tokenizers.BertWordPieceTokenizer}
    indexer_map = {'plain_lyrics': PlainLyricsIndexer, 'structured_lyrics':StructuredLyricsIndexer}
    args=parser.parse_args()
    tokenizer_class = tokenizers_map[args.tokenizer_class]
    subencoder_class = subencoder_map[args.subencoder_class]
    indexer_class = indexer_map[args.indexer_type]
    indexer = indexer_class(args.encoder_class, args.use_morphs,  vocab_size=args.vocab_size,
                            tokenizer_class=tokenizer_class, encoder_class=subencoder_class, jamo=args.split_jamo)
    print(args.directory_path, args.inp_path)
    indexer.corpus_encode(args.directory_path, args.inp_path, args.out_path)


if __name__ == "__main__":
    main()

