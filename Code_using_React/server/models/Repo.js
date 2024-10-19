import mongoose from 'mongoose';

const repoSchema = new mongoose.Schema({
  githubUrl: {
    type: String,
    required: true,
  },
  projectName: {
    type: String,
    required: true,
  },
  repoName: {
    type: String,
    required: true,
  },
});

export default mongoose.model('Repo', repoSchema);
