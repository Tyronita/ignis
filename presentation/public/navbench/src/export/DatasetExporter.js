// Downloads episode data as JSON in the shape of a real trajectory dataset
// file: per-timestep {timestamp, position, heading, velocity,
// angular_velocity, collided}, with the static per-episode facts (seed, room,
// obstacle layout) attached once rather than repeated on every row.
export class DatasetExporter {
  static downloadEpisode(episodeData) {
    this._download(episodeData, `episode_${episodeData.episodeIndex}_seed${episodeData.room_seed}.json`);
  }

  static downloadSession(episodes) {
    this._download({ episodeCount: episodes.length, episodes }, `session_${Date.now()}.json`);
  }

  static _download(obj, filename) {
    const blob = new Blob([JSON.stringify(obj)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoking immediately races the browser actually starting to read the blob —
    // harmless for small files, but silently drops larger downloads (observed with
    // multi-episode session exports). Deferring it is the standard fix.
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
}
