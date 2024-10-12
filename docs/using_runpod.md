# Runpod

We have an account in runpod.io and we get an account with the team email and the password(check Discord for that).

I have uploaded the data to a network storage volume called 'raw_data'.

## Run training using runpod

1. Install `runpodctl` using `brew install runpod/tap/runpodctl`
2. `runpodctl config --apiKey <see-discord>`
3. Create a pod using `runpodctl create pod --name team5-training --templateId pytorch --networkVolumeId raw_data`

## Log into the pods using ssh

You need to create your own ssh keys and add the public key to runpod. This way you should be able to ssh into the pods.

The instructions are [here](https://docs.runpod.io/pods/connect-to-a-pod)
