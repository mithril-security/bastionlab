# Proto Files

## Proto Files Structure

<pre>
<b>protos 🚀🔐/</b>
├── <a href="#bastionlab.proto">bastionlab.proto</a>
├── <a href="#bastionlab_polars.proto">bastionlab_polars.proto</a>
└── <a href="#bastionlab_torch.proto">bastionlab_torch.proto</a>
</pre>



## bastionlab.proto

### Messages

* #### **Empty** 

  `{}`

* #### **ChallengeResponse**

  `{ bytes value = 1; }`

* #### **SessionInfo**

  `{ bytes token = 1; expiry_time = 2 }`

* #### **ClientInfo**

  | Name               | Type     | Value |
  | ------------------ | -------- | ----- |
  | uid                | `string` | 1     |
  | platform_name      | `string` | 2     |
  | platform_arch      | `string` | 3     |
  | platform_version   | `string` | 4     |
  | platform_release   | `string` | 5     |
  | user_agent         | `string` | 6     |
  | user_agent_version | `string` | 7     |
  | is_colab           | `bool`   | 8     |

## bastionlab_polars.proto

### Messages

* #### **Empty**

  `{}`

* #### **ReferenceRequest**

  `{ string identifier = 1; }`

* #### **ReferenceResponse**

  `{ string identifier = 1; string header = 2 }`

* #### **ReferenceList**

  `{ repeated ReferenceResponse list = 1; }`

* #### **Query**

   `{ string composite_plan = 1 }`

* #### SendChunk

  | Name     | Type     | Value |
  | -------- | -------- | ----- |
  | data     | `bytes`  | 1     |
  | policy   | `string` | 2     |
  | metadata | `string` | 3     |

* #### FetchChunk

  **oneof** body:

  | Name    | Type     | Value |
  | ------- | -------- | ----- |
  | data    | `bytes`  | 1     |
  | pending | `string` | 2     |
  | warning | `string` | 3     |

## bastionlab_torch.proto

### Messages

* #### **Empty**

  `{}`

* #### Accuracy

  `{ float value = 1; }`

* #### Devices

  `{ repeated string list = 1; }`

* #### Optimizers

  `{ repeated string list = 1; }`

* #### Metric

  | Name        | Type    | Value |
  | ----------- | ------- | ----- |
  | value       | `float` | 1     |
  | uncertainty | `float` | 2     |
  | batch       | `int32` | 3     |
  | epoch       | `int32` | 4     |
  | nb_epochs   | `int32` | 5     |
  | nb_batches  | `int32` | 6     |

* #### Reference

  | Name        | Type     | Value |
  | ----------- | -------- | ----- |
  | identifier  | `string` | 1     |
  | name        | `string` | 2     |
  | description | `string` | 3     |
  | meta        | `bytes`  | 4     |

* #### References

  `{ repeated Reference list = 1; }`

* #### TestConfig

  | Name       | Type        | Value |
  | ---------- | ----------- | ----- |
  | model      | `Reference` | 1     |
  | dataset    | `Reference` | 2     |
  | batch_size | `int32`     | 3     |
  | device     | `string`    | 4     |
  | metric     | `string`    | 5     |
  | metric_eps | `float`     | 6     |

* #### Chunk

  | Name        | Type     | Value |
  | ----------- | -------- | ----- |
  | data        | `bytes`  | 1     |
  | name        | `string` | 2     |
  | description | `string` | 3     |
  | secret      | `bytes`  | 4     |
  | meta        | `bytes`  | 5     |

* #### TrainConfig

  | Name                    | Type        | Value |
  | ----------------------- | ----------- | ----- |
  | model                   | `Reference` | 1     |
  | dataset                 | `Reference` | 2     |
  | batch_size              | `int32`     | 3     |
  | epochs                  | `int32`     | 4     |
  | device                  | `string`    | 5     |
  | metric                  | `string`    | 6     |
  | eps                     | `float`     | 7     |
  | max_grad_norm           | `float`     | 8     |
  | metric_eps              | `float`     | 9     |
  | per_n_steps_checkpoint  | `int32`     | 12    |
  | per_n_epochs_checkpoint | `int32`     | 13    |
  | resume                  | `bool`      | 14    |

  * **oneof** optimizer

    `{ SGD sgd = 10; Adam adam = 11; }`

  * **message SGD**

    | Name         | Type    | Value |
    | ------------ | ------- | ----- |
    | larning_rate | `float` | 1     |
    | weight_decay | `float` | 2     |
    | momentum     | `float` | 3     |
    | dampening    | `float` | 4     |
    | nesterov     | `bool`  | 5     |

  * **message Adam**

    | Name         | Type    | Value |
    | ------------ | ------- | ----- |
    | larning_rate | `float` | 1     |
    | beta_1       | `float` | 2     |
    | beta_2       | `float` | 3     |
    | epsilon      | `float` | 4     |
    | weight_decay | `float` | 5     |
    | amsgrad      | `float` | 6     |

    