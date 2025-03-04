library(netmeta)
help(Senn2013)
data(Senn2013)
data15 <- Senn2013
help(netmeta)
args(netmeta)
mn1 <- netmeta(TE, seTE, treat1, treat2, studlab, data = data15, ref = "plac", sm = "MD")
mn1$TE.common
mn1$seTE.common
mn1$upper.common
mn1$lower.common

library(feather)
width_95 <- mn1$upper.common - mn1$lower.common
write_feather(as.data.frame(width_95), "./tmp/ci_width.feather")
