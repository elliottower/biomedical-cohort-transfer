#!/usr/bin/env Rscript
# Export IBD data from curatedMetagenomicData to TSV files.
#
# Usage: Rscript batch2_extensions/ibd_microbiome/export_cmd_data.R
#
# Outputs:
#   batch2_extensions/ibd_microbiome/data/species_profiles.tsv
#   batch2_extensions/ibd_microbiome/data/metadata.tsv

library(curatedMetagenomicData)

outdir <- "batch2_extensions/ibd_microbiome/data"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

cat("Querying available IBD studies...\n")

# Get all available datasets
all_datasets <- curatedMetagenomicData("*relative_abundance", dryrun = TRUE)

# Filter for IBD-related studies
ibd_conditions <- c("IBD", "CD", "UC")

# Get sample metadata
smd <- sampleMetadata
ibd_samples <- smd[smd$study_condition %in% ibd_conditions | smd$study_condition == "control", ]

# Get studies that have IBD samples
ibd_studies <- unique(ibd_samples$study_name[ibd_samples$study_condition %in% ibd_conditions])
cat(sprintf("Found %d studies with IBD samples\n", length(ibd_studies)))

# Filter to studies with enough samples
study_counts <- table(ibd_samples$study_name[ibd_samples$study_name %in% ibd_studies])
good_studies <- names(study_counts[study_counts >= 30])
cat(sprintf("Studies with >= 30 samples: %d\n", length(good_studies)))

# Get samples from good studies (IBD + controls)
keep_samples <- ibd_samples[ibd_samples$study_name %in% good_studies, ]
keep_samples <- keep_samples[keep_samples$study_condition %in% c(ibd_conditions, "control"), ]
cat(sprintf("Total samples: %d\n", nrow(keep_samples)))

# Print study breakdown
for (study in good_studies) {
  sub <- keep_samples[keep_samples$study_name == study, ]
  n_ibd <- sum(sub$study_condition %in% ibd_conditions)
  n_ctrl <- sum(sub$study_condition == "control")
  cat(sprintf("  %s: %d IBD, %d control\n", study, n_ibd, n_ctrl))
}

# Download species-level relative abundance
cat("\nDownloading species profiles...\n")
dataset_names <- paste0(good_studies, ".relative_abundance")
dataset_names <- dataset_names[dataset_names %in% all_datasets]

if (length(dataset_names) == 0) {
  cat("No matching datasets found. Trying alternative query...\n")
  # Try fetching via returnSamples
  se <- returnSamples(keep_samples, "relative_abundance")
  abundance <- as.data.frame(assay(se))
  meta_out <- as.data.frame(colData(se))
} else {
  results <- curatedMetagenomicData(dataset_names, dryrun = FALSE)
  # Merge all into one
  if (is(results, "list")) {
    se <- do.call(cbind, results)
  } else {
    se <- results
  }

  # Filter to our samples
  common <- intersect(colnames(se), rownames(keep_samples))
  se <- se[, common]
  abundance <- as.data.frame(assay(se))
  meta_out <- as.data.frame(colData(se))
}

# Write outputs
cat(sprintf("\nWriting %d features x %d samples...\n", nrow(abundance), ncol(abundance)))
write.table(abundance, file.path(outdir, "species_profiles.tsv"),
            sep = "\t", quote = FALSE)

meta_export <- data.frame(
  Sample_ID = rownames(meta_out),
  Study = meta_out$study_name,
  Group = ifelse(meta_out$study_condition %in% ibd_conditions, "IBD", "control"),
  Condition = meta_out$study_condition,
  stringsAsFactors = FALSE
)
write.table(meta_export, file.path(outdir, "metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat(sprintf("Done. %d studies, %d samples\n", length(good_studies), ncol(abundance)))
