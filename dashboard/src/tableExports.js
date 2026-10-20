export function summaryCsv(rows) {
  const columns = ['cell_line', 'candidate_site_records', 'median_reads_per_site', 'percent_sites_1_2_reads', 'percent_sites_20_plus_reads', 'scope'];
  const quote = value => `"${String(value).replaceAll('"', '""')}"`;
  return [columns.join(','), ...rows.map(row => [row.line, row.sites, row.median_reads, row.pct_1_2, row.pct_20_plus, 'Full pooled dataset; all included transcripts'].map(quote).join(','))].join('\r\n');
}

export function exportSummaryCsv(rows) {
  const url = URL.createObjectURL(new Blob([summaryCsv(rows)], {type: 'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'full-pooled-read-support.csv';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
