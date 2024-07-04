# install.packages("devtools")
# devtools::install_github("vasishth/lingpsych")

library("lingpsych")

# prepare Hindi dataframe
keep_cols_hi <- c("subj", "expt", "session", "item", "lang", "roi", "FFD", "FFP", 
                  "SFD", "FPRT", "RBRT", "TFT", "RPD", "CRPD", "RRT", "RRTP", 
                  "RRTR", "RBRC", "TRC", "LPRT", "word_lex", "word_len")
df_hi_selected_cols <- df_hindi_full[keep_cols_hi]
# get only the first session because in the second people will have already
# seen the same text but in Urdu in the first session
df_hi <- df_hi_selected_cols[df_hi_selected_cols$session == 1, ]
df_hi <- df_hi[df_hi$expt != 'prac', ]
colnames(df_hi) <- c("participant", "expt", "session", "sent_id_and_idx", "lang", "word_idx", 
                     "FFD", "FFP", "SFD", "FPRT", "RBRT", "TFT", "RPD", "CRPD", "RRT", "RRTP", 
                     "RRTR", "RBRC", "TRC", "LPRT","word", "word_len")

keep_cols_hi2 <- c("participant", "sent_id_and_idx", "lang", "word_idx","FFD", "FFP", 
                   "SFD", "FPRT", "RBRT", "TFT", "RPD", "CRPD", "RRT", "RRTP", 
                   "RRTR", "RBRC", "TRC", "LPRT", "word", "word_len")
df_hi_final <- df_hi[keep_cols_hi2]
df_hi_final$entropy <- sample(100, size = nrow(df_hi_final), replace = TRUE)
df_hi_final$surprisal <- sample(100, size = nrow(df_hi_final), replace = TRUE)
df_hi_final <- df_hi_final[df_hi_final$word != '।',]
df_hi_final <- df_hi_final[with(df_hi_final, order(participant, sent_id_and_idx, word_idx)), ]

# add total_word_idx as column for easy merging with RNN output
total_word_idx <- c()
participant <- 1
sent <- 1
word <- 0

for (i in 1:nrow(df_hi_final)){
  row <- df_hi_final[i,]
  
  current_participant <- row$participant
  current_sent <- row$sent_id_and_idx
  current_word <- row$word_idx
  
  if (current_participant == participant){
    if (current_sent == sent){
      word <- word + 1
      total_word_idx <- append(total_word_idx, word)
    } else if (current_sent > sent){
      sent <- current_sent
      word <- word + 1
      total_word_idx <- append(total_word_idx, word)
    } else {
      print(c(participant, current_sent, sent))
      break
    }
  } else if (current_participant > participant){
    participant <- current_participant
    sent <- 1
    word <- 1
    total_word_idx <- append(total_word_idx, word)
  }
}
df_hi_final$total_word_idx <- total_word_idx

write.csv(df_hi_final, file="df_hi_potsdam.csv", row.names=FALSE)

test <- read.csv("df_hi_example_struct_v3.csv")


# prepare English dataframe
load(file='joint_data_trimmed.rda')
df_en <- joint.data[joint.data$lang == 'en', ]
keep_cols_en <- c("uniform_id", "trialid", "sentnum", "ianum", 
                  "nrun", "reread", "nfix", "refix", "reg.in", "dur", 
                  "firstrun.skip", "firstrun.nfix", "firstrun.refix", 
                  "firstrun.reg.in", "firstrun.dur", "firstrun.gopast", 
                  "firstrun.gopast.sel", "firstfix.sac.in", "firstfix.sac.out", 
                  "firstfix.launch", "firstfix.land", "firstfix.cland", 
                  "firstfix.dur", "singlefix", "ia")
df_en <- df_en[keep_cols_en]
# uniform_id == participant, ia == word, trialid + sentnum == sent_id_and_idx, lang == en,
# ianum == word_idx, create word_len
df_en$participant <- as.numeric(factor(df_en$uniform_id))
# sort the dataframe so that the for loop works
df_en <- df_en[with(df_en, order(participant, trialid, sentnum)), ]
df_en <- head(df_en, -1) # remove NA row

sent_id_and_idx_en <- c()
word_idx_en <- c()
word_lens <- c()
participant <- 1
trial <- 1
sent_idx <- 1
sent_in_text <- 1
word_idx <- 0


for (i in 1:nrow(df_en)){
  row <- df_en[i,]
  current_participant <- row$participant
  current_trial <- row$trialid
  current_sent_idx <- row$sentnum
  
  current_word <- gsub('[[:punct:] ]+','',row$ia)
  word_lens <- append(word_lens, nchar(current_word))
  
  if (current_participant == participant){
    if (current_sent_idx == sent_idx){
      sent_id_and_idx_en <- append(sent_id_and_idx_en, sent_in_text)
      word_idx <- word_idx + 1
      word_idx_en <- append(word_idx_en, word_idx)
      
    } else{
      sent_in_text <- sent_in_text + 1
      sent_idx <- current_sent_idx
      sent_id_and_idx_en <- append(sent_id_and_idx_en, sent_in_text)
      word_idx <- 1
      word_idx_en <- append(word_idx_en, word_idx)
    }

  } else if (current_participant == participant + 1){
    participant <- current_participant
    trial <- current_trial
    sent_idx <- 1
    sent_in_text <- 1
    word_idx <- 1
    sent_id_and_idx_en <- append(sent_id_and_idx_en, sent_in_text)
    word_idx_en <- append(word_idx_en, word_idx)
  }
  
}

df_en$sent_id_and_idx <- sent_id_and_idx_en
df_en$word_idx <- word_idx_en
df_en$word_len <- word_lens
df_en$lang <- rep("English", nrow(df_en))

keep_cols_en2 <- c("participant", "trialid", "sent_id_and_idx", "lang", "word_idx",
                   "nrun", "reread", "nfix", "refix", "reg.in", "dur", 
                   "firstrun.skip", "firstrun.nfix", "firstrun.refix", 
                   "firstrun.reg.in", "firstrun.dur", "firstrun.gopast", 
                   "firstrun.gopast.sel", "firstfix.sac.in", "firstfix.sac.out", 
                   "firstfix.launch", "firstfix.land", "firstfix.cland", 
                   "firstfix.dur", "singlefix", "ia", "word_len", "ianum")
df_en_final <- df_en[keep_cols_en2]
colnames(df_en_final) <- c("participant", "text_id", "sent_id_and_idx", "lang", 
                           "word_idx", "nrun", "reread", "nfix", "refix", "reg.in", "dur", 
                           "firstrun.skip", "firstrun.nfix", "firstrun.refix", 
                           "firstrun.reg.in", "firstrun.dur", "firstrun.gopast", 
                           "firstrun.gopast.sel", "firstfix.sac.in", "firstfix.sac.out", 
                           "firstfix.launch", "firstfix.land", "firstfix.cland", 
                           "firstfix.dur", "singlefix","word", "word_len", "total_word_idx")
df_en_final[df_en_final == ""] <- NA  
df_en_final <- df_en_final[!is.na(df_en_final$word),]
write.csv(df_en_final, file="df_en_meco.csv", row.names=FALSE)
test2 <- read.csv("df_en_meco.csv")

# get all the MECO text as a list of strings 
words_meco <- read.csv("supp texts.csv")
words_meco_en <- words_meco[words_meco$X == 'English', ]
words_meco_en
words_en <- as.character(as.vector(words_meco_en))
write(words_en, "words_en_meco.txt")


# get all the Potsdam H1 text
words_hindi1 <- read.csv("hnd1_aswords_stats.csv", sep=" ")
sentences_h1 <- list()
item_id <- 1

words <- c()
for(i in 1:nrow(words_hindi1)) {
  row <- words_hindi1[i,]
  current_item <- row$item
  word <- row$word_lex
  
  if (current_item == item_id){
    words <- append(words, word)
    
  } else {
    item_id <- current_item
    sentences_h1[[item_id - 1]] <- paste(words, collapse=" ")
    words <- c(word)
  }
}
sentences_h1[[item_id]] <- paste(words, collapse=" ")

lapply(sentences_h1, write, "hindi1_words.txt", append=TRUE)

# get all the Potsdam H2 text
words_hindi2 <- read.csv("h2_fix.csv", sep=";", fill=TRUE)
sentences_h2 <- list()
item_id <- 1

words <- c()
for(i in 1:nrow(words_hindi2)) {
  row <- words_hindi2[i,]
  current_item <- row$item
  word <- row$word_lex
  
  if (current_item == item_id){
    words <- append(words, word)
    
  } else {
    item_id <- current_item
    sentences_h2[[item_id - 1]] <- paste(words, collapse=" ")
    words <- c(word)
  }
}
sentences_h2[[item_id]] <- paste(words, collapse=" ")

lapply(sentences_h2, write, "hindi2_words.txt", append=TRUE)


