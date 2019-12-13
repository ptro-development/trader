# !!! Some initial warnings !!!
This project was very hacky from a start, so do not expect any nice code as most of it was about proving if some idea woked or not. The project took over year with the discarding a lot of prototypes. However, if it may be interesting to look at as those prototypes used thigs like probability trees for trading sequnces, euclidian multispace choise selction for investmens, neuron networks driven by genetic algorithms to do the best prediction for the investment, GPU code for search correlation amoung masive amount of samples ... etc .You have been warrned :)       

# Trader project
This project was created to play with gathered trading bitcoin data collected over few months trought a websoket in high resolution.

# Idea
The initial idea was to split the data into smaller chunks of data sequences (let's us call them trading sentences or samples). Then each sentence consisted of multiple consequent parts where every part was an avarage of some time period.

Example of sentence:

[(avarage of firts 5 minutes),(avarage of next 5 minutes), (avarage of next 5 minutes) ...]

The sentence had to have reasonable gain or loss between the start and end of trading sequence otherwise there were discarded.

Next code randomly sampled collected trading data for the chosen amount of sentences. Then the whole trading data was walked to find correlations which those chosen samples (NxM problem) The most occuring samples were stored and less occuring discarded. 

(Technical note: Initially this was written by using Cellery message broker to do processing over multiple CPU cores as it was very computational heavy. Yet with changing granularity / resolution of sampled sentences and their ammont this quikly went of controle. 8 CPU cores and around 50 000 of samples took just 5 days to compute :) Only SciPy libraries and Cellery were used at this point. So, next iterration was to rewrite the most CPU heavy code to RUN on GPU. You can find code in open_cl directory. This took some time reverse engineer SciPy libs for GPU. I used open_cl due to some good speed up results for gentic algorithms I wrote in Java and C in past. Python multiprocess module was used as well do some quick multi core processing where it still made sense. So finally few GBytes for data processing and aroud 1500 GPU cores later   computation took "only" few hours :)

Ok, at this point I got samples / sentences which were used to build probability trading trees. There were actually built two trees for gain and loss. The most occuring nodes in tree lead way the highest probability for loss or gain when trading.

In next step a multilaer back-propagation neural network with genetic algorithm was trained to walk trading data let say to first two months to get best gain and smalest loss.

(Technical note: I have chosen existing python library implementing network where I had an easy access to its weights. The network did not "know" GA so I wrote one and pluged that in to manipulate its internal weights. Generally above probability trees were chosen when deciding how much to invest when simmilar / known strat of sequence was spotted. This process of traning was quite slow as the network ran only one CPU core. Later rewritten to run at least on multiple CPU cores.)

After the training followed simulated trading on next (unknown data for algorithm) month of trading data. The code was able to get some money :) Hurruy unfortunatelly not much :)

So, it was followed by implementing different strategies and their implemntation. The most successfull was the euclidian multispace choise selction for investmens which was actually quite easy to implement.

To be continued ...










