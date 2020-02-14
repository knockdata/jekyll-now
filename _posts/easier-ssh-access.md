# Simplify SSH access with SSH config 

SSH is one of the most common ways to access remote machines.
It is very helpful if we don't have remember and type all those command line options.

Here is a short instruction of using ssh config to simplify daily life.

## Generate SSH key pairs

There are different key types, RSA is the default type on openssh.
A new type, ED25519 is the most recommend public-key algorithm available today. 

Here are short reason, why ED25519 is recommend. For more detail please refer to this [medium article](https://medium.com/risan/upgrade-your-ssh-key-to-ed25519-c6e8d60d3c54)

* ED25519 is much short (readable) for the same level of security 
* any random number can be an Ed25519 key. To generate an RSA you have to generate two large random primes, and the code that does this is complicated an so can more easily be (and in the past has been) compromised to generate weak keys.

While not all system supported ED25519 keys, like add as the default public key when creating Azure VM.

So I generate both RSA and ED25519 keys. 
Use ED25519 if the system support it. 
Otherwise fall back to RSA

Generate ED25519 key pair. 

    ssh-keygen -t ed25519 -C "{your_name}@example.com"
       
Generate RSA key pair
 
    ssh-keygen -C "{your_name}@example.com" 

Two notes when create SSH key. 

* Add -C "{your_name}@example.com"  to easy identify the key pair who belong to
* Using a passphrase to encrypt the key. So even people get the private key, it will not create any harm. The breach will only happen when both the private key and phassphrase stolen. 

## Add ssh key to the ssh agent.
 
ssh agent can help to remember the passphrase and decrypt the private key when using ssh command line.

First ensure ssh agent is running.

    $ eval "$(ssh-agent -s)"
    > Agent pid {some_number}

Then add the ssh key, you will asked to provide passphrase when run the following command.

    ssh-add -K ~/.ssh/id_ed25519
    ssh-add -K ~/.ssh/id_rsa


## General ~/.ssh/config

A common sector can be added in ssh config file. Here is an example for most command setup.

    Host *
      IgnoreUnknown UseKeychain
      User rockie
      AddKeysToAgent yes
      UseKeychain yes
      ForwardAgent yes
      IdentityFile ~/.ssh/id_ed25519
      IdentityFile ~/.ssh/id_rsa
       
* User: it can be used if this is the most common user logon to systems.
* UseKeychain yes: This will use MacOS key chain. 
* IgnoreUnknown UseKeychain: It will ignore the unsupported instruction UseKeychain yes on Linux.
* ForwardAgent yes: This is useful when have proxy
* IdentityFile: Multiple private keys can be added with precedence. 

New sections can be added for each machine want to access. 
The instruction will override the instruction in common part

    Host nd19
        User rockie
        Hostname 172.238.44.19  

Now we can ssh to 172.238.44.19 with just command

    ssh nd19
             
## Proxy setup

ssh config support tunneling with ProxyCommand instruction. Here is a short example.

    Host proxy
        Hostname 172.238.44.19  
        
    Host edge
        Hostname 10.238.46.112
        ProxyCommand ssh -q -W %h:%p proxy

In this case we will access machine edge 10.238.46.112 through proxy machine 172.238.44.19.


        
