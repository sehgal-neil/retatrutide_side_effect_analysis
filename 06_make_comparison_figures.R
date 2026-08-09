#!/usr/bin/env Rscript

# Exploratory grouped bar charts comparing retatrutide with the prior
# semaglutide/tirzepatide Reddit study at PT, HLT, and HLGT levels.
# Uses base R only.

`%||%` <- function(a, b) if (!is.null(a)) a else b

cmd_args <- commandArgs(FALSE)
file_arg <- cmd_args[grepl("^--file=", cmd_args)][1] %||% NA
if (!is.na(file_arg)) {
  script_path <- sub("^--file=", "", file_arg)
  project_dir <- dirname(normalizePath(script_path))
} else {
  project_dir <- getwd()
}

outputs_dir <- file.path(project_dir, "outputs")
figures_dir <- file.path(project_dir, "figures")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)

glp1_denominator <- 29172
reta_denominator <- 7823

glp1_hlgt <- data.frame(
  term = c(
    "Gastrointestinal signs and symptoms",
    "Gastrointestinal motility and defaecation conditions",
    "General system disorders NEC",
    "Appetite and general nutritional disorders",
    "Neurological disorders NEC",
    "Headaches",
    "Skin appendage conditions",
    "Anxiety disorders and symptoms",
    "Sleep disorders and disturbances",
    "Depressed mood disorders and disturbances",
    "Epidermal and dermal conditions",
    "Muscle disorders",
    "Menstrual cycle and uterine bleeding disorders",
    "Gastrointestinal conditions NEC",
    "Physical examination and organ system status topics",
    "Musculoskeletal and connective tissue disorders NEC",
    "Mood disorders and disturbances NEC",
    "Electrolyte and fluid balance conditions",
    "Administration site reactions",
    "Glucose metabolism disorders (incl diabetes mellitus)",
    "Cardiac arrhythmias"
  ),
  n = c(15879, 8576, 7833, 4196, 3555, 2031, 1532, 1421, 1140, 1051,
        1005, 992, 923, 894, 721, 696, 684, 630, 619, 614, 562),
  percent = c(54.4, 29.4, 26.9, 14.4, 12.2, 7.0, 5.3, 4.9, 3.9, 3.6,
              3.4, 3.4, 3.2, 3.1, 2.5, 2.4, 2.3, 2.2, 2.1, 2.1, 1.9),
  stringsAsFactors = FALSE
)

glp1_hlt <- data.frame(
  term = c(
    "Nausea and vomiting symptoms",
    "Gastrointestinal atonic and hypomotility disorders NEC",
    "Asthenic conditions",
    "Appetite disorders",
    "Diarrhea (excl infective)",
    "Gastrointestinal and abdominal pains (excl oral and throat)",
    "Dyspeptic signs and symptoms",
    "Flatulence, bloating and distension",
    "Headaches NEC",
    "Feelings and sensations NEC",
    "Neurological signs and symptoms NEC",
    "Gastrointestinal signs and symptoms NEC",
    "Anxiety symptoms",
    "Disturbances in consciousness NEC",
    "Alopecias",
    "Disturbances in initiating and maintaining sleep",
    "Gastrointestinal disorders NEC",
    "Pain and discomfort NEC",
    "Depressive disorders",
    "Sensory abnormalities NEC",
    "Physical examination procedures and organ system status",
    "Menstruation and uterine bleeding NEC",
    "Musculoskeletal and connective tissue pain and discomfort",
    "Injection site reactions",
    "Hypoglycaemic conditions NEC",
    "Rate and rhythm disorders NEC",
    "Muscle pains",
    "Paraesthesias and dysaesthesias",
    "Total fluid volume decreased",
    "Apocrine and eccrine gland disorders",
    "Muscle related signs and symptoms NEC",
    "General nutritional disorders NEC",
    "Emotional and mood disturbances NEC",
    "Oral dryness and saliva altered",
    "Dermal and epidermal conditions NEC",
    "General signs and symptoms NEC",
    "Joint related signs and symptoms",
    "Pruritus NEC",
    "Tremor (excl congenital)",
    "Mood alterations with depressive symptoms"
  ),
  n = c(12690, 5946, 5677, 3979, 3687, 2601, 2551, 2227, 1789, 1689,
        1508, 1310, 1288, 1109, 918, 908, 891, 856, 827, 753, 721, 710,
        641, 619, 602, 548, 544, 495, 473, 420, 395, 392, 383, 362,
        350, 347, 341, 335, 333, 323),
  percent = c(43.5, 20.4, 19.5, 13.6, 12.6, 8.9, 8.7, 7.6, 6.1, 5.8,
              5.2, 4.5, 4.4, 3.8, 3.1, 3.1, 3.1, 2.9, 2.8, 2.6, 2.5,
              2.4, 2.2, 2.1, 2.1, 1.9, 1.9, 1.7, 1.6, 1.4, 1.4, 1.3,
              1.3, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1),
  stringsAsFactors = FALSE
)

glp1_pt <- data.frame(
  term = c(
    "Nausea", "Fatigue", "Vomiting", "Constipation", "Diarrhoea",
    "Decreased appetite", "Abdominal pain", "Eructation",
    "Gastrooesophageal reflux disease", "Headache"
  ),
  n = c(10764, 4870, 4749, 4463, 3686, 3371, 2493, 2010, 1877, 1784),
  percent = c(36.9, 16.7, 16.3, 15.3, 12.6, 11.6, 8.5, 6.9, 6.4, 6.1),
  stringsAsFactors = FALSE
)

read_reta <- function(level) {
  if (level == "hlgt") {
    d <- read.csv(file.path(outputs_dir, "table_hlgt_counts_appetite_combined.csv"), stringsAsFactors = FALSE)
    d <- d[, c("hlgt_term", "n_users", "percent_users")]
  } else if (level == "hlt") {
    d <- read.csv(file.path(outputs_dir, "table_hlt_counts_appetite_combined.csv"), stringsAsFactors = FALSE)
    d <- d[, c("hlt_term", "n_users", "percent_users")]
  } else if (level == "pt") {
    d <- read.csv(file.path(outputs_dir, "table_pt_by_soc_appetite_combined.csv"), stringsAsFactors = FALSE)
    d <- d[, c("pt_term", "n_users", "percent_users")]
  } else {
    stop("Unknown level: ", level)
  }
  names(d) <- c("term", "Retatrutide_n", "Retatrutide_percent")
  d <- d[order(-d$Retatrutide_percent), ]
  d
}

get_glp1 <- function(level) {
  if (level == "hlgt") glp1_hlgt else if (level == "hlt") glp1_hlt else if (level == "pt") glp1_pt else stop(level)
}

wrap_label <- function(x, width) {
  vapply(strwrap(x, width = width, simplify = FALSE), paste, collapse = "\n", FUN.VALUE = character(1))
}

make_data <- function(level, top_n = 10) {
  reta <- read_reta(level)
  data <- reta[seq_len(min(top_n, nrow(reta))), ]
  glp <- get_glp1(level)
  data$GLP1_n <- glp$n[match(data$term, glp$term)]
  data$GLP1_percent <- glp$percent[match(data$term, glp$term)]
  data$GLP1_n[is.na(data$GLP1_n)] <- 0
  data$GLP1_percent[is.na(data$GLP1_percent)] <- 0
  data
}

draw_plot <- function(
  data,
  level,
  title,
  out_prefix,
  label_width = 18,
  png_width = 2600,
  png_height = 1500,
  pdf_width = 12,
  pdf_height = 7,
  cex_names = 0.68,
  cex_axis = 0.9,
  cex_legend = 0.9,
  cex_value_labels = 0.62,
  cex_main = 1.0,
  cex_lab = 1.0,
  cex_footnote = 0.72,
  bottom_mar = 10,
  left_mar = 5,
  top_mar = 3,
  right_mar = 1,
  x_label_srt = 90,
  x_label_y_offset = 0.03,
  x_label_adj = 1,
  show_value_labels = TRUE,
  footnote_line = 8.7
) {
  data_file <- file.path(figures_dir, paste0(out_prefix, "_data.csv"))
  png_file <- file.path(figures_dir, paste0(out_prefix, ".png"))
  pdf_file <- file.path(figures_dir, paste0(out_prefix, ".pdf"))
  eps_file <- file.path(figures_dir, paste0(out_prefix, ".eps"))
  write.csv(data, data_file, row.names = FALSE)

  draw <- function(file, device) {
    if (device == "png") png(file, width = png_width, height = png_height, res = 220)
    if (device == "pdf") pdf(file, width = pdf_width, height = pdf_height)
    if (device == "eps") {
      postscript(
        file,
        width = pdf_width,
        height = pdf_height,
        horizontal = FALSE,
        onefile = FALSE,
        paper = "special"
      )
    }
    op <- par(no.readonly = TRUE)
    on.exit({ par(op); dev.off() }, add = TRUE)
    par(mar = c(bottom_mar, left_mar, top_mar, right_mar), xpd = FALSE)
    vals <- rbind(
      Retatrutide = data$Retatrutide_percent,
      `Semaglutide/tirzepatide` = data$GLP1_percent
    )
    ymax <- max(vals) * 1.15
    bp <- barplot(
      vals,
      beside = TRUE,
      ylim = c(0, ymax),
      col = c("#0072B2", "#D55E00"),
      border = NA,
      axes = FALSE,
      ylab = "Users with symptom term/group, %",
      cex.axis = cex_axis,
      cex.lab = cex_lab,
      cex.main = cex_main,
      main = title
    )
    axis(2, las = 1, cex.axis = cex_axis)
    box(bty = "l")
    grid(nx = NA, ny = NULL, col = "gray88", lty = 1)
    label_x <- if (is.matrix(bp)) colMeans(bp) else bp
    label_y <- par("usr")[3] - ymax * x_label_y_offset
    text(
      x = label_x,
      y = label_y,
      labels = wrap_label(data$term, label_width),
      srt = x_label_srt,
      adj = c(x_label_adj, 1),
      cex = cex_names,
      xpd = NA
    )
    legend(
      "topright",
      legend = c("Retatrutide", "Semaglutide/tirzepatide"),
      fill = c("#0072B2", "#D55E00"),
      border = NA,
      bty = "n",
      cex = cex_legend,
      x.intersp = 0.9,
      y.intersp = 1.1
    )
    if (show_value_labels) {
      labels <- ifelse(vals == 0, "<0.5", sprintf("%.1f", vals))
      text(x = bp, y = vals + ymax * 0.02, labels = labels, cex = cex_value_labels, srt = 90, adj = 0)
    }
    mtext(
      paste0("Top retatrutide ", toupper(level), "s after combining appetite-related PTs; ordered by retatrutide percentage."),
      side = 1,
      line = footnote_line,
      cex = cex_footnote
    )
  }

  draw(png_file, "png")
  draw(pdf_file, "pdf")
  draw(eps_file, "eps")
  cat("Wrote", data_file, "\n")
  cat("Wrote", png_file, "\n")
  cat("Wrote", pdf_file, "\n")
  cat("Wrote", eps_file, "\n")
}

draw_plot(
  make_data("hlgt"),
  "hlgt",
  "HLGT Symptom Groups Discussed by Reddit Users",
  "hlgt_comparison_bar_chart",
  label_width = 18
)

draw_plot(
  make_data("hlt"),
  "hlt",
  "MedDRA HLT Symptom Groups Reported by Reddit Users",
  "hlt_comparison_bar_chart",
  label_width = 22,
  png_width = 3000,
  png_height = 2200,
  pdf_width = 14,
  pdf_height = 10,
  cex_names = 1.2,
  cex_axis = 1.45,
  cex_legend = 1.32,
  cex_value_labels = 1.0,
  cex_main = 1.5,
  cex_lab = 1.4,
  cex_footnote = 0.9,
  bottom_mar = 22,
  left_mar = 6,
  top_mar = 4,
  x_label_srt = 55,
  x_label_y_offset = 0.13,
  x_label_adj = 1,
  show_value_labels = FALSE,
  footnote_line = 20.4
)

draw_plot(
  make_data("pt"),
  "pt",
  "Preferred Terms Discussed by Reddit Users",
  "pt_comparison_bar_chart",
  label_width = 18
)
