#!/usr/bin/env Rscript --no-save

# Imports BED-files corresponding to entries in metadata.tsv (file generated when downloading batches of samples from encode)
# Computes genome-wide enrichments based on overlaps with peaks in 1kb tiles
# Exports enrichments to hdf5 file
# Exports corresponding sequences to FASTA files

library(dplyr)
library(data.table)
library(GenomicAlignments)
library(rtracklayer)
library(matrixStats)

printlog<-function(...){
cat(format(Sys.time(), "%y-%m-%d %X   "),paste(...),'\n')
}

md<-read.delim('metadata_filtered.tsv')
md<-arrange(md, Experiment.accession)

md<-group_by(md, Experiment.accession)
md<-mutate(md, qrel = n_peaks/max(n_peaks))

mdf<-filter(md, n_peaks >= 1000)
mdf<-filter(mdf, qrel >= 0.8) 

bedfiles<-paste0(mdf$File.accession, '.bed')
write.table(mdf, 'metadata_filtered_processed.tsv', quote=F, row.names=F, sep='\t')
md<-mdf

printlog('importing BED-files...')
beds<-lapply(bedfiles, function(x){
    printlog("importing",x)
    return(fread(x)) } 
)

si<-Seqinfo(genome='hg19')

toGR<-function(x){
	return(GRanges(seqnames=x$V1, IRanges(x$V2, x$V3), strand=Rle('*',nrow(x)), score=x$V7, seqinfo=si))
}

for ( i in seq_along(beds) ){
    # printlog(i,'out of',length(beds)) 
    beds[[i]] <- toGR(beds[[i]])
}
printlog('done.')

beds <- GRangesList(beds)
beds <- keepStandardChromosomes(beds, pruning.mode="tidy")

printlog('generating genome-wide tiles...')
tiles <- tileGenome(tilewidth = 1000, seqlengths = seqlengths(si))
tiles<-unlist(tiles)
tiles<-keepStandardChromosomes(tiles, pruning.mode="tidy")
printlog('done.')

get_enrichments<-function(t, p){
    s <- matrix(1,nrow=length(t),ncol=length(p))
    for ( i in seq_along(p) ){
        printlog('calculating enrichments using',length(p[[i]]),'peaks...')
        olaps<-findOverlaps(t,p[[i]],minoverlap = 50)
        ol_t<-data.table(data.frame(olaps))
        setnames(ol_t, c('qh','sh'))
        ol_t$width <- width(overlapsRanges(ranges(t),ranges(p[[i]]),olaps))
        ol_t$score <- p[[i]]$score[ol_t$sh]
        ol_t <- ol_t[,list(score=sum(score*width/1000)+((1000-sum(width))/1000)),by=qh]
        s[ ol_t$qh,i ] <- ol_t$score
    }
    return(rowSums2(s)/ncol(s))
}

accessions<-as.character(unique(md$Experiment.accession))
groups<-lapply(accessions, function(x){which(md$Experiment.accession == x)})

overlap_mat_scores<-vector('list',length=length(groups))

printlog('calculating enrichments...')
for ( i in seq_along(groups)){
    printlog('group',i,'out of',length(groups))
    overlap_mat_scores[[i]]<-get_enrichments(tiles, beds[ groups[[i]] ])
    names(overlap_mat_scores)[i]<-accessions[i]
}
printlog('done.')

overlap_mat_scores<-do.call(cbind, overlap_mat_scores)

attr(overlap_mat_scores,'colID')<-accessions
seq_ids<-paste0(seqnames(tiles),':',start(tiles),'-',end(tiles))
attr(overlap_mat_scores,'seqID')<-seq_ids
dimnames(overlap_mat_scores)<-NULL

printlog('exporting enrichments...')
library(rhdf5)
# fwrite(data.frame(overlap_mat_scores), file='overlaps_scores.csv', quote = F, sep = ',', row.names = T)
if (file.exists('gw_enrichments.h5')) file.remove('gw_enrichments.h5')
h5createFile("gw_enrichments.h5")
h5createGroup("gw_enrichments.h5","data")
h5createDataset("gw_enrichments.h5","data/enrichments",dims=c(dim(overlap_mat_scores)[1],dim(overlap_mat_scores)[2]),storage.mode=storage.mode(overlap_mat_scores),chunk=c(10000,1),level=4)
h5createGroup("gw_enrichments.h5","meta")
h5createDataset(file="gw_enrichments.h5","meta/colID",dims=length(accessions), storage.mode="character",size=max(nchar(accessions)))
h5createDataset(file="gw_enrichments.h5","meta/seqID",dims=length(seq_ids), storage.mode="character",size=max(nchar(seq_ids)),chunk=10000)
h5write(overlap_mat_scores,"gw_enrichments.h5","data/enrichments")
h5write(seq_ids, "gw_enrichments.h5","meta/seqID")
h5write(accessions, "gw_enrichments.h5","meta/colID")
printlog('done.')

library(BSgenome.Hsapiens.UCSC.hg19)
library(Biostrings)
printlog('retrieving tile sequences and writing to sequences[_rt].fa...')
seqs<-getSeq(BSgenome.Hsapiens.UCSC.hg19, tiles)
names(seqs)<-seq_ids
writeXStringSet(seqs, filepath='sequences.fa')
seqs<-reverseComplement(seqs)
writeXStringSet(seqs, filepath='sequences_rc.fa')
printlog("done.")
