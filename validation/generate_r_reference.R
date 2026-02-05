# Generate reference data from R ENMeval for Python parity testing
# Run with: Rscript validation/generate_r_reference.R

.libPaths(c("~/R/library", .libPaths()))

library(ENMeval)

# Load built-in example data
data(bvariegatus)

cat("Loaded bvariegatus:", nrow(bvariegatus), "occurrence records\n")
cat("Columns:", paste(colnames(bvariegatus), collapse=", "), "\n")

# Export the occurrence data for Python
write.csv(bvariegatus, "validation/bvariegatus_occs.csv", row.names=FALSE)

cat("\nOccurrence data saved to validation/bvariegatus_occs.csv\n")
