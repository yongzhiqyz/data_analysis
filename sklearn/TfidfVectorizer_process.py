from sklearn.feature_extraction.text import TfidfVectorizer
"""
 the TfidfVectorizer aims to deal with text
"""

corpus = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
# corpus = [u'这是第一个文档.',
#       u'这是第二个文档.',
#       u'这是第三个.',
#       u'这是第一个文档吗?',]

vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(vectorizer.fit_transform(corpus).toarray())
