# !!! Some initial warnings !!!
This project was very hacky from a start, so do not expect any nice code as most of it was about proving if some idea worked or not. That would be a phase two if successful. The project took over year with the discarding a lot of prototypes and I mean really a lot. However, if it may be interesting to look at as those prototypes which used things like probability trees for trading sequences, euclidean multispace choice selection for investments, neuron networks driven by genetic algorithms to do the best prediction for the investment, GPU code for search correlation among massive amount of samples ... etc .You have been warned :)      

# Trader project
This project was created to play with gathered trading bitcoin data collected over few months trough a websoket in high resolution.

# Idea
The initial idea was to split the data into smaller chunks of data sequences (let's us call them trading sentences or samples). Then each sentence consisted of multiple consequent parts where every part was an average of some time period.

Example of sentence:

[(average of first 5 minutes),(average of next 5 minutes), (average of next 5 minutes) ...]

The sentence had to have reasonable gain or loss between the start and end of trading sequence otherwise there were discarded.

Next code randomly sampled collected trading data for the chosen amount of sentences. Then the whole trading data was walked to find correlations which those chosen samples (N x M problem) The most occurring samples were stored and less occurring discarded.

(Technical note: Initially this was written by using Cellery message broker to do processing over multiple CPU cores as it was very computational heavy. Yet with changing granularity / resolution of sampled sentences and their amount this quickly went of control. 8 CPU cores and around 50 000 of samples took just 5 days to compute :) Only SciPy libraries and Celery were used at this point. So, next iteration was to rewrite the most CPU heavy code to RUN on GPU. You can find code in open_cl directory. This took some time reverse engineer SciPy libs for GPU. I used open_cl due to some good speed up results for genetic algorithms I wrote in Java and C in past. Python multiprocess module was used as well to do some quick multicore processing where it still made sense. So finally few GBytes for data processing and around 1500 GPU cores later computation took "only" few hours :)

Ok, at this point I got samples / sentences which were used to build probability trading trees. There were actually built two trees for a gain and loss. The most occurring nodes in tree lead way the highest probability for loss or gain when trading.

In next step a multilayer back-propagation neural network with genetic algorithm was trained to walk trading data let say to first two months to get best gain and smallest loss.

(Technical note: I have chosen existing python library implementing network where I had an easy access to its weights. The network did not "know" GA so I wrote one and plugged that in to manipulate its internal weights. Generally above probability trees were chosen when deciding how much to invest when similar / known start of sequence was spotted. This process of training was quite slow as the network ran only one CPU core. Later it was rewritten to run at least on multiple CPU cores.)

After the training followed simulated trading on next (unknown data for algorithm) month of trading data. The code was able to get some money :) Hurrah, unfortunately not so much :)

So, it was followed by implementing different strategies and their implementation. The most successful was the euclidean multispace choice selection for investments which was actually quite easy to implement.

# Learned lessons

- data is a king, so make sure that your data is complete. I had few gaps in trading data when the collection over the websocket went down. I did not think that it had a big impact on project yet as soon as you start doing a faster trading (few seconds long sentences) it was a problem.

- analyze your data before you start coding, this may help you to answer some basic questions and narrow down a problem space you are addressing. In better case it tells to not even start working on project :) If you are lucky you may find some probability distribution at least. For example, it helped a lot to do statistical analysis on relations among the length of sampled sequence, trading gain and their count. As it turned out there were some exponential relations which helped with the choice of length sampled sequence and targeted gain. Some chaotic systems have their attractors / edges. (e.g.: measles outbreaks, you can see some "edges" even on butterfly weather system). Then a subtle signal may indicate that you are approaching the edge. I tried to analyze the data by available tools for chaos series to spot any possible attractors yet without success. More a more analysis I did only pointed to the system without patterns and memory (which is bad news for the neural networks and generally predictability). 

- test the simplest ideas first, they take less time to prototype. For example following seemed to work the best.

-- avoiding the loss by adding loss probability tree into the trading worked better than just focusing on the trading gain

-- euclidean multispace choice selection for investments  when trading was much more successful than the neural network with GA yet combination of both gave better result

- as it usually is with a hard project as this one a final gain is in technology you learn on the way rather than the end result of project. So, I have learned the new technologies for parallel processing of data and tools for analysis.

# Not tested

- I have not tested to find any correlation between trade markets and social media which could be an interesting project.














