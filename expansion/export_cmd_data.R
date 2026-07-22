#!/usr/bin/env Rscript
# Export CRC and T2D species-level relative abundances from curatedMetagenomicData.
#
# Install once:
#   install.packages("BiocManager")
#   BiocManager::install("curatedMetagenomicData")
#
# Run:
#   /usr/local/opt/r/bin/Rscript expansion/export_cmd_data.R
#
# Outputs:
#   expansion/data/crc_species.tsv      (species x samples)
#   expansion/data/crc_metadata.tsv     (samples x fields)
#   expansion/data/t2d_species.tsv
#   expansion/data/t2d_metadata.tsv

library(curatedMetagenomicData)

dir.create("expansion/data", recursive = TRUE, showWarnings = FALSE)

export_disease <- function(conditions, out_prefix, body_site = "stool") {
    cat(sprintf("Querying sampleMetadata for conditions: %s\n",
                paste(conditions, collapse = ", ")))

    md <- sampleMetadata
    md <- md[!is.na(md$study_name) & !is.na(md$study_condition), ]

    studies_with_disease <- unique(md$study_name[md$study_condition %in% conditions])
    md_filt <- md[md$study_name %in% studies_with_disease &
                  (md$study_condition %in% conditions | md$study_condition == "control") &
                  md$body_site == body_site, ]

    cat(sprintf("  Found %d samples across %d studies\n",
                nrow(md_filt), length(unique(md_filt$study_name))))
    cat(sprintf("  Studies: %s\n",
                paste(unique(md_filt$study_name), collapse = ", ")))

    for (s in sort(unique(md_filt$study_name))) {
        n_case <- sum(md_filt$study_name == s & md_filt$study_condition %in% conditions)
        n_ctrl <- sum(md_filt$study_name == s & md_filt$study_condition == "control")
        cat(sprintf("    %s: %d case, %d control\n", s, n_case, n_ctrl))
    }

    # curatedMetagenomicData expects a single regex pattern per dataset
    study_names <- unique(md_filt$study_name)
    pattern <- paste0("(", paste(study_names, collapse = "|"), ").relative_abundance")
    cat(sprintf("  Fetching with pattern: %s\n", pattern))

    se_list <- curatedMetagenomicData(pattern, dryrun = FALSE, counts = FALSE)

    if (!is(se_list, "list")) {
        se_list <- list(se_list)
    }

    # Merge assay matrices manually to avoid rowData column conflicts
    all_features <- unique(unlist(lapply(se_list, rownames)))
    meta_frames <- lapply(se_list, function(se) as.data.frame(colData(se)))
    common_cols <- Reduce(intersect, lapply(meta_frames, colnames))
    merged_meta <- do.call(rbind, lapply(meta_frames, function(df) df[, common_cols, drop = FALSE]))
    uid <- rownames(merged_meta)
    merged_mat <- matrix(0, nrow = length(all_features), ncol = length(uid),
                         dimnames = list(all_features, uid))
    for (nm in names(se_list)) {
        se <- se_list[[nm]]
        se_uid <- paste0(nm, ".", rownames(colData(se)))
        merged_mat[rownames(se), se_uid] <- as.matrix(assay(se))
    }

    keep <- which(merged_meta$study_name %in% study_names &
                  (merged_meta$study_condition %in% conditions |
                   merged_meta$study_condition == "control") &
                  !is.na(merged_meta$body_site) &
                  merged_meta$body_site == body_site)
    merged_mat <- merged_mat[, keep]
    merged_meta <- merged_meta[keep, , drop = FALSE]
    cat(sprintf("  Merged: %d features x %d samples\n", nrow(merged_mat), ncol(merged_mat)))

    species_mat <- as.data.frame(merged_mat)
    meta_out <- merged_meta
    meta_out$is_case <- as.integer(meta_out$study_condition %in% conditions)

    species_file <- paste0(out_prefix, "_species.tsv")
    meta_file <- paste0(out_prefix, "_metadata.tsv")

    write.table(species_mat, species_file, sep = "\t", quote = FALSE)
    write.table(meta_out, meta_file, sep = "\t", quote = FALSE)

    cat(sprintf("  Wrote %s (%d x %d)\n", species_file, nrow(species_mat), ncol(species_mat)))
    cat(sprintf("  Wrote %s (%d samples)\n\n", meta_file, nrow(meta_out)))
}

export_disease(c("CRC"),
               "expansion/data/crc")

export_disease(c("T2D"),
               "expansion/data/t2d")
