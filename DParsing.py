from nltk.parse.corenlp import CoreNLPDependencyParser

# # Path to CoreNLP jar unzipped
# jar_path = '/content/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar'

# # Path to CoreNLP model jar
# models_jar_path = '/content/stanford-corenlp-4.2.2-models-english.jar'

sentence = 'Deemed universities charge huge fees.'

# Initialize StanfordDependency Parser from the path
# parser = StanfordDependencyParser(path_to_jar = jar_path, path_to_models_jar = models_jar_path)
parser = CoreNLPDependencyParser(url="http://nlp.stanford.edu:8080")

# Parse the sentence
result = parser.raw_parse(sentence)
dependency = result.__next__()


print ("{:<15} | {:<10} | {:<10} | {:<15} | {:<10}".format('Head', 'Head POS','Relation','Dependent', 'Dependent POS'))
print ("-" * 75)
  
# Use dependency.triples() to extract the dependency triples in the form
# ((head word, head POS), relation, (dependent word, dependent POS))  
for dep in list(dependency.triples()):
  print ("{:<15} | {:<10} | {:<10} | {:<15} | {:<10}"
         .format(str(dep[0][0]),str(dep[0][1]), str(dep[1]), str(dep[2][0]),str(dep[2][1])))